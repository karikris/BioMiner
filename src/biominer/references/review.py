from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
from datetime import UTC, date, datetime
import errno
import fcntl
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

import polars as pl

from biominer.references.deduplication import (
    validate_reference_media_deduplication_artifacts,
)
from biominer.references.licensing import ReferenceLicencePolicy
from biominer.references.schemas import (
    REFERENCE_LIFE_STAGES,
    REFERENCE_REVIEW_DECISIONS_SCHEMA_VERSION,
    REFERENCE_REVIEW_QUEUE_SCHEMA_VERSION,
    REFERENCE_VIEWS,
    REFERENCE_VISUAL_DOMAINS,
    make_reference_review_decision_id,
    make_reference_review_request_id,
    reference_acquisition_selection_schema,
    reference_acquisition_selections_frame,
    reference_media_candidate_schema,
    reference_media_candidates_frame,
    reference_media_duplicate_relationship_schema,
    reference_media_duplicate_relationships_frame,
    reference_media_object_schema,
    reference_media_objects_frame,
    reference_observation_schema,
    reference_observations_frame,
    reference_review_decisions_frame,
    reference_review_queue_frame,
    reference_review_queue_schema,
    validate_reference_review_decisions,
    validate_reference_review_queue,
    write_reference_review_decisions,
    write_reference_review_queue,
)
from biominer.reports.flickr_fetch import current_git_sha
from biominer.storage.parquet import write_parquet


REFERENCE_REVIEW_DECISION_IMPORT_SCHEMA_VERSION = (
    "reference-review-decision-import-v1.0.0"
)
REFERENCE_REVIEW_OUTCOMES_SCHEMA_VERSION = "reference-review-outcomes-v1.0.0"
REFERENCE_REVIEW_CONFLICTS_SCHEMA_VERSION = "reference-review-conflicts-v1.0.0"
REFERENCE_REVIEW_RESOLVED_MEDIA_SCHEMA_VERSION = (
    "reference-review-resolved-media-v1.0.0"
)
REFERENCE_REVIEW_QUEUE_PROVENANCE_SCHEMA_VERSION = (
    "reference-review-queue-provenance-v1.0.0"
)
REFERENCE_REVIEW_EXPORT_REPORT_VERSION = "reference-review-export-report-v1.0.0"
REFERENCE_REVIEW_IMPORT_REPORT_VERSION = "reference-review-import-report-v1.0.0"
REFERENCE_REVIEW_HISTORY_SCHEMA_VERSION = "reference-review-history-v1.0.0"
REFERENCE_REVIEW_HISTORY_HEAD_SCHEMA_VERSION = "reference-review-history-head-v1.0.0"

REFERENCE_REVIEW_DECISION_TEMPLATE_FILE = "reference_review_decision_template.parquet"
REFERENCE_REVIEW_DECISION_IMPORT_FILE = "reference_review_decision_import.parquet"
REFERENCE_REVIEW_DECISIONS_FILE = "reference_review_decisions.parquet"
REFERENCE_REVIEW_QUEUE_PROVENANCE_FILE = "reference_review_queue_provenance.parquet"
REFERENCE_REVIEW_OUTCOMES_FILE = "reference_review_outcomes.parquet"
REFERENCE_REVIEW_CONFLICTS_FILE = "reference_review_conflicts.parquet"
REFERENCE_REVIEW_VERIFIED_FILE = "verified_reference_media.parquet"
REFERENCE_REVIEW_EXCLUDED_FILE = "excluded_reference_media.parquet"
REFERENCE_REVIEW_EXPORT_REPORT_FILE = "reference_review_export_report.json"
REFERENCE_REVIEW_EXPORT_SUMMARY_FILE = "reference_review_export_summary.md"
REFERENCE_REVIEW_IMPORT_REPORT_FILE = "reference_review_import_report.json"
REFERENCE_REVIEW_IMPORT_SUMMARY_FILE = "reference_review_import_summary.md"

_DECISION_STATUSES = frozenset({"verified", "excluded", "uncertain"})
_REVIEW_CONFIDENCE_VALUES = frozenset({"high", "medium", "low", "unknown"})
_PRODUCTION_VISUAL_DOMAINS = frozenset({"live_field", "pinned_specimen"})
_REVIEWER_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._:@/-]{2,127}\Z")
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HISTORY_ID_PATTERN = re.compile(r"reference-review-history:[0-9a-f]{64}\Z")
_LICENCE_POLICY = ReferenceLicencePolicy()
_DECISION_IMPORT_SORT = [
    "reference_media_id",
    "review_round",
    "verified_by",
    "reviewed_at",
    "review_request_id",
]
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferenceReviewQueueResult:
    queue: pl.DataFrame
    provenance: pl.DataFrame
    decision_template: pl.DataFrame
    report: dict[str, Any]
    markdown: str


@dataclass(frozen=True)
class ReferenceReviewWorkflowResult:
    queue: pl.DataFrame
    provenance: pl.DataFrame
    decision_import: pl.DataFrame
    decisions: pl.DataFrame
    outcomes: pl.DataFrame
    conflicts: pl.DataFrame
    verified: pl.DataFrame
    excluded: pl.DataFrame
    report: dict[str, Any]
    markdown: str


def reference_review_decision_import_schema() -> dict[str, pl.DataType]:
    return {
        "import_schema_version": pl.String,
        "review_request_id": pl.String,
        "reference_media_id": pl.String,
        "review_round": pl.UInt16,
        "verified_by": pl.String,
        "reviewed_at": pl.Datetime("us", "UTC"),
        "target_identity_verified": pl.Boolean,
        "verification_status": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "view": pl.String,
        "review_confidence": pl.String,
        "review_notes": pl.String,
        "exclusion_reason": pl.String,
        "conflicts_with_decision_id": pl.String,
    }


def reference_review_queue_provenance_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "review_request_id": pl.String,
        "reference_media_id": pl.String,
        "source_binding_fingerprint": pl.String,
        "source_leaf_fingerprints": pl.List(pl.String),
        "queue_semantics_fingerprint": pl.String,
        "queue_row_fingerprint": pl.String,
        "input_fingerprint": pl.String,
    }


def reference_review_outcome_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "review_request_id": pl.String,
        "reference_media_id": pl.String,
        "review_status": pl.String,
        "effective_decision_ids": pl.List(pl.String),
        "effective_reviewer_ids": pl.List(pl.String),
        "decisive_reviewer_count": pl.UInt16,
        "required_review_count": pl.UInt8,
        "resolved_verification_status": pl.String,
        "target_identity_verified": pl.Boolean,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "view": pl.String,
        "second_review_required": pl.Boolean,
        "support_eligible": pl.Boolean,
        "blocker_reasons": pl.List(pl.String),
        "queue_fingerprint": pl.String,
        "decision_ledger_fingerprint": pl.String,
        "resolved_at": pl.Datetime("us", "UTC"),
    }


def reference_review_conflict_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "conflict_group_id": pl.String,
        "review_request_id": pl.String,
        "reference_media_id": pl.String,
        "effective_decision_ids": pl.List(pl.String),
        "effective_reviewer_ids": pl.List(pl.String),
        "conflicting_fields": pl.List(pl.String),
        "resolution_status": pl.String,
        "detected_at": pl.Datetime("us", "UTC"),
    }


def reference_review_resolved_media_schema() -> dict[str, pl.DataType]:
    schema = {
        "schema_version": pl.String,
        **{
            field: dtype
            for field, dtype in reference_review_queue_schema().items()
            if field != "schema_version"
        },
    }
    schema.update(
        {
            "resolved_verification_status": pl.String,
            "target_identity_verified": pl.Boolean,
            "resolved_life_stage": pl.String,
            "resolved_visual_domain": pl.String,
            "resolved_view": pl.String,
            "effective_decision_ids": pl.List(pl.String),
            "effective_reviewer_ids": pl.List(pl.String),
            "resolved_exclusion_reasons": pl.List(pl.String),
        }
    )
    return schema


def build_reference_review_queue(
    selections: pl.DataFrame,
    media_objects: pl.DataFrame,
    media_candidates: pl.DataFrame,
    observations: pl.DataFrame,
    duplicate_relationships: pl.DataFrame,
    *,
    deduplication_report: Mapping[str, object],
    reference_bank_version: str,
    created_at: datetime | None = None,
    include_research_only: bool = False,
) -> ReferenceReviewQueueResult:
    run_started_at = datetime.now(UTC)
    timestamp = _utc_datetime(
        created_at
        if created_at is not None
        else _default_queue_created_at(selections, media_objects),
        field="created_at",
    )
    bank_version = _required_canonical_text(
        reference_bank_version,
        field="reference_bank_version",
    )
    context = {
        "command": "references.export_review_queue",
        "reference_bank_version": bank_version,
        "created_at": timestamp.isoformat(),
    }
    _log_event("reference_review_queue_started", **context)
    try:
        result = _build_reference_review_queue(
            selections,
            media_objects,
            media_candidates,
            observations,
            duplicate_relationships,
            deduplication_report=deduplication_report,
            reference_bank_version=bank_version,
            created_at=timestamp,
            include_research_only=include_research_only,
            report_started_at=run_started_at,
        )
    except Exception as exc:
        _log_event(
            "reference_review_queue_failed",
            **context,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    _log_event(
        "reference_review_queue_completed",
        **context,
        counts=result.report["counts"],
    )
    return result


def validate_reference_review_queue_source_bindings(
    queue: pl.DataFrame,
    provenance: pl.DataFrame,
    selections: pl.DataFrame,
    media_objects: pl.DataFrame,
    media_candidates: pl.DataFrame,
    observations: pl.DataFrame,
    duplicate_relationships: pl.DataFrame,
    *,
    deduplication_report: Mapping[str, object],
    reference_bank_version: str,
) -> None:
    """Rebuild immutable review inputs and reject stale manual decisions."""

    validate_reference_review_queue(queue)
    _validate_queue_integrity(queue, provenance)
    created_at_values = set(queue["created_at"].to_list())
    if len(created_at_values) > 1:
        raise ValueError("reference review queue has inconsistent creation times")
    created_at = (
        next(iter(created_at_values))
        if created_at_values
        else _default_queue_created_at(selections, media_objects)
    )
    include_research_only = bool(
        not queue.is_empty()
        and queue.filter(pl.col("licence_policy_status") == "research_only").height
    )
    rebuilt = _build_reference_review_queue(
        selections,
        media_objects,
        media_candidates,
        observations,
        duplicate_relationships,
        deduplication_report=deduplication_report,
        reference_bank_version=reference_bank_version,
        created_at=_utc_datetime(created_at, field="created_at"),
        include_research_only=include_research_only,
        report_started_at=datetime.now(UTC),
    )
    current_queue = {
        str(row["reference_media_id"]): row for row in queue.iter_rows(named=True)
    }
    rebuilt_queue = {
        str(row["reference_media_id"]): row
        for row in rebuilt.queue.iter_rows(named=True)
    }
    if set(current_queue) != set(rebuilt_queue):
        raise ValueError("reference review queue does not match current source inventory")
    for media_id, row in current_queue.items():
        expected = rebuilt_queue[media_id]
        immutable_fields = (
            "review_request_id",
            "input_fingerprint",
            "media_object_fingerprint",
            "durable_preview_uri",
        )
        if any(row[field] != expected[field] for field in immutable_fields) or (
            _queue_semantics_fingerprint(row)
            != _queue_semantics_fingerprint(expected)
        ):
            raise ValueError(
                "reference review queue is stale for current source inventory: "
                + media_id
            )

    current_provenance = {
        str(row["reference_media_id"]): row
        for row in provenance.iter_rows(named=True)
    }
    rebuilt_provenance = {
        str(row["reference_media_id"]): row
        for row in rebuilt.provenance.iter_rows(named=True)
    }
    for media_id, row in current_provenance.items():
        expected = rebuilt_provenance[media_id]
        immutable_fields = (
            "review_request_id",
            "source_binding_fingerprint",
            "source_leaf_fingerprints",
            "queue_semantics_fingerprint",
            "input_fingerprint",
        )
        if any(row[field] != expected[field] for field in immutable_fields):
            raise ValueError(
                "reference review source provenance is stale for current inventory: "
                + media_id
            )


def _build_reference_review_queue(
    selections: pl.DataFrame,
    media_objects: pl.DataFrame,
    media_candidates: pl.DataFrame,
    observations: pl.DataFrame,
    duplicate_relationships: pl.DataFrame,
    *,
    deduplication_report: Mapping[str, object],
    reference_bank_version: str,
    created_at: datetime,
    include_research_only: bool,
    report_started_at: datetime,
) -> ReferenceReviewQueueResult:
    selections = _normalize_frame(
        selections,
        schema=reference_acquisition_selection_schema(),
        constructor=reference_acquisition_selections_frame,
        artifact="reference acquisition selections",
    )
    media_objects = _normalize_frame(
        media_objects,
        schema=reference_media_object_schema(),
        constructor=reference_media_objects_frame,
        artifact="reference media objects",
    )
    media_candidates = _normalize_frame(
        media_candidates,
        schema=reference_media_candidate_schema(),
        constructor=reference_media_candidates_frame,
        artifact="reference media candidates",
    )
    observations = _normalize_frame(
        observations,
        schema=reference_observation_schema(),
        constructor=reference_observations_frame,
        artifact="reference observations",
    )
    duplicate_relationships = _normalize_frame(
        duplicate_relationships,
        schema=reference_media_duplicate_relationship_schema(),
        constructor=reference_media_duplicate_relationships_frame,
        artifact="reference media duplicate relationships",
    )
    validate_reference_media_deduplication_artifacts(
        media_objects=media_objects,
        relationships=duplicate_relationships,
        media_candidates=media_candidates,
        observations=observations,
        report=deduplication_report,
    )

    object_by_id = _unique_rows(media_objects, "reference_media_id")
    candidate_by_id = _unique_rows(media_candidates, "reference_media_id")
    observation_by_id = _unique_rows(observations, "reference_observation_id")
    _validate_inventory_foreign_keys(
        object_by_id=object_by_id,
        candidate_by_id=candidate_by_id,
        observation_by_id=observation_by_id,
    )
    relationships_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for relationship in duplicate_relationships.iter_rows(named=True):
        group_id = str(relationship["duplicate_group_id"])
        relationships_by_group[group_id].append(dict(relationship))
        _validate_duplicate_relationship_foreign_keys(
            relationship,
            object_by_id=object_by_id,
            candidate_by_id=candidate_by_id,
        )
    _validate_duplicate_relationship_completeness(
        media_objects,
        relationships_by_group,
    )
    member_ids_by_group: dict[str, list[str]] = defaultdict(list)
    for obj in object_by_id.values():
        if obj["decode_status"] == "valid":
            member_ids_by_group[str(obj["duplicate_group_id"])].append(
                str(obj["reference_media_id"])
            )

    selected_contexts: list[dict[str, object]] = []
    skipped_reasons: Counter[str] = Counter()
    for selection in selections.iter_rows(named=True):
        media_id = str(selection["reference_media_id"])
        candidate = candidate_by_id.get(media_id)
        if candidate is None:
            raise ValueError(f"selected media {media_id} has no media candidate")
        obj = object_by_id.get(media_id)
        if obj is None:
            raise ValueError(f"selected media {media_id} has no media object")
        observation_id = str(selection["reference_observation_id"])
        observation = observation_by_id.get(observation_id)
        if observation is None:
            raise ValueError(
                f"selected media {media_id} has no source observation {observation_id}"
            )
        _validate_selection_provenance(selection, candidate, observation, obj)
        exclusion = _queue_exclusion_reason(
            candidate,
            obj,
            include_research_only=include_research_only,
        )
        if exclusion is not None:
            skipped_reasons[exclusion] += 1
            continue
        group_id = str(obj["duplicate_group_id"])
        group_relationships = relationships_by_group.get(group_id, [])
        _validate_duplicate_group_taxonomy(
            member_ids_by_group[group_id],
            candidate_by_id=candidate_by_id,
            observation_by_id=observation_by_id,
        )
        unresolved = any(
            str(row["resolution_status"]) != "resolved" for row in group_relationships
        )
        queue_media_id = (
            media_id if unresolved else str(obj["canonical_reference_media_id"])
        )
        queue_object = object_by_id.get(queue_media_id)
        queue_candidate = candidate_by_id.get(queue_media_id)
        if queue_object is None or queue_candidate is None:
            raise ValueError(
                f"duplicate group {group_id} canonical media inventory is incomplete"
            )
        queue_observation = observation_by_id.get(
            str(queue_candidate["reference_observation_id"])
        )
        if queue_observation is None:
            raise ValueError(f"queue media {queue_media_id} has no source observation")
        _validate_media_provenance(queue_candidate, queue_observation, queue_object)
        canonical_exclusion = _queue_exclusion_reason(
            queue_candidate,
            queue_object,
            include_research_only=include_research_only,
        )
        if canonical_exclusion is not None:
            raise ValueError(
                "canonical queue media is not review eligible: "
                f"{queue_media_id} ({canonical_exclusion})"
            )
        selected_contexts.append(
            {
                "selection": dict(selection),
                "selected_object": dict(obj),
                "selected_candidate": dict(candidate),
                "selected_observation": dict(observation),
                "queue_media_id": queue_media_id,
                "queue_object": dict(queue_object),
                "queue_candidate": dict(queue_candidate),
                "queue_observation": dict(queue_observation),
                "group_members": tuple(
                    (
                        dict(object_by_id[group_media_id]),
                        dict(candidate_by_id[group_media_id]),
                        dict(
                            observation_by_id[
                                str(
                                    candidate_by_id[group_media_id][
                                        "reference_observation_id"
                                    ]
                                )
                            ]
                        ),
                    )
                    for group_media_id in sorted(member_ids_by_group[group_id])
                ),
                "relationships": tuple(group_relationships),
                "unresolved_duplicate": unresolved,
            }
        )

    contexts_by_media: dict[str, list[dict[str, object]]] = defaultdict(list)
    for context in selected_contexts:
        contexts_by_media[str(context["queue_media_id"])].append(context)

    queue_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    for queue_media_id, contexts in sorted(contexts_by_media.items()):
        queue_row, provenance_row = _queue_row(
            queue_media_id,
            contexts,
            reference_bank_version=reference_bank_version,
            created_at=created_at,
        )
        queue_rows.append(queue_row)
        provenance_rows.append(provenance_row)
    queue = reference_review_queue_frame(queue_rows)
    provenance = _strict_frame(
        provenance_rows,
        schema=reference_review_queue_provenance_schema(),
        sort_by=["reference_media_id", "review_request_id"],
    )
    _validate_queue_integrity(queue, provenance)
    template = reference_review_decision_template(queue, provenance)
    empty_decisions = reference_review_decisions_frame([])
    report_ended_at = datetime.now(UTC)
    report = {
        "schema_version": REFERENCE_REVIEW_EXPORT_REPORT_VERSION,
        "command": "references.export_review_queue",
        "status": "complete",
        "run_id": "not_instrumented",
        "pid": os.getpid(),
        "git_sha": current_git_sha(),
        "started_at": report_started_at.isoformat(),
        "ended_at": report_ended_at.isoformat(),
        "elapsed_seconds": max(
            0.0,
            (report_ended_at - report_started_at).total_seconds(),
        ),
        "queue_created_at": created_at.isoformat(),
        "reference_bank_version": reference_bank_version,
        "settings": {"include_research_only": include_research_only},
        "history": _review_history_record(
            queue,
            provenance,
            empty_decisions,
            revision=0,
            parent_report_sha256=None,
            new_decision_ids=(),
        ),
        "inputs": {
            "selection_rows": selections.height,
            "media_object_rows": media_objects.height,
            "media_candidate_rows": media_candidates.height,
            "observation_rows": observations.height,
            "duplicate_relationship_rows": duplicate_relationships.height,
            "selections_fingerprint": _frame_fingerprint(selections),
            "media_objects_fingerprint": _frame_fingerprint(media_objects),
            "media_candidates_fingerprint": _frame_fingerprint(media_candidates),
            "observations_fingerprint": _frame_fingerprint(observations),
            "duplicate_relationships_fingerprint": _frame_fingerprint(
                duplicate_relationships
            ),
            "deduplication_report_fingerprint": _payload_hash(deduplication_report),
        },
        "counts": {
            "selected_rows": selections.height,
            "queue_rows": queue.height,
            "queue_provenance_rows": provenance.height,
            "skipped_rows": sum(skipped_reasons.values()),
            "collapsed_selected_rows": max(
                0,
                len(selected_contexts) - queue.height,
            ),
            "research_only_rows": queue.filter(
                pl.col("licence_policy_status") == "research_only"
            ).height,
        },
        "skipped_reason_counts": dict(sorted(skipped_reasons.items())),
        "outputs": {
            "queue_fingerprint": _frame_fingerprint(queue),
            "queue_provenance_fingerprint": _frame_fingerprint(provenance),
            "decision_template_fingerprint": _frame_fingerprint(template),
            "artifact_uris": "not_instrumented",
        },
        "metrics": {
            "api_calls": None,
            "retries": None,
            "rows_per_second": None,
            "peak_rss_bytes": "not_instrumented",
        },
    }
    return ReferenceReviewQueueResult(
        queue=queue,
        provenance=provenance,
        decision_template=template,
        report=report,
        markdown=_review_markdown(report, title="Reference review queue export"),
    )


def reference_review_decision_template(
    queue: pl.DataFrame,
    provenance: pl.DataFrame,
) -> pl.DataFrame:
    validate_reference_review_queue(queue)
    _validate_queue_integrity(queue, provenance)
    rows = [
        {
            "import_schema_version": REFERENCE_REVIEW_DECISION_IMPORT_SCHEMA_VERSION,
            "review_request_id": row["review_request_id"],
            "reference_media_id": row["reference_media_id"],
            "review_round": 1,
            "verified_by": None,
            "reviewed_at": None,
            "target_identity_verified": None,
            "verification_status": None,
            "life_stage": row["life_stage"],
            "visual_domain": row["visual_domain"],
            "view": row["view"],
            "review_confidence": None,
            "review_notes": None,
            "exclusion_reason": None,
            "conflicts_with_decision_id": None,
        }
        for row in queue.iter_rows(named=True)
    ]
    return _strict_frame(
        rows,
        schema=reference_review_decision_import_schema(),
        sort_by=_DECISION_IMPORT_SORT,
    )


def import_reference_review_decisions(
    raw_decisions: pl.DataFrame,
    *,
    queue: pl.DataFrame,
    queue_provenance: pl.DataFrame,
    existing_decisions: pl.DataFrame,
    prior_report: Mapping[str, object],
    prior_report_sha256: str,
    resolved_at: datetime | None = None,
) -> ReferenceReviewWorkflowResult:
    run_started_at = datetime.now(UTC)
    event_context = {
        "command": "references.import_review_decisions",
        "raw_decision_rows": getattr(raw_decisions, "height", None),
        "queue_rows": getattr(queue, "height", None),
        "existing_decision_rows": getattr(existing_decisions, "height", 0),
    }
    _log_event("reference_review_import_started", **event_context)
    try:
        _validate_import_frame(raw_decisions)
        validate_reference_review_queue(queue)
        _validate_queue_integrity(queue, queue_provenance)
        existing = existing_decisions
        validate_reference_review_decisions(existing)
        _validate_decision_source_hashes(existing)
        _validate_prior_review_projection(queue, queue_provenance, existing)
        prior_history = _validate_prior_review_report(
            queue,
            queue_provenance,
            existing,
            prior_report=prior_report,
            prior_report_sha256=prior_report_sha256,
        )

        imported_rows = [
            _normalize_raw_decision(row) for row in raw_decisions.iter_rows(named=True)
        ]
        merged, imported_count, replay_count = _merge_decision_rows(
            existing,
            imported_rows,
        )
        _validate_decision_source_hashes(merged)
        result = resolve_reference_review_statuses(
            queue,
            merged,
            queue_provenance=queue_provenance,
            resolved_at=resolved_at,
        )
        run_ended_at = datetime.now(UTC)
        existing_ids = set(existing["review_decision_id"])
        new_decision_ids = sorted(
            set(result.decisions["review_decision_id"]) - existing_ids
        )
        report = json.loads(json.dumps(result.report))
        report.update(
            {
                "schema_version": REFERENCE_REVIEW_IMPORT_REPORT_VERSION,
                "command": "references.import_review_decisions",
                "started_at": run_started_at.isoformat(),
                "ended_at": run_ended_at.isoformat(),
                "elapsed_seconds": max(
                    0.0,
                    (run_ended_at - run_started_at).total_seconds(),
                ),
                "history": _review_history_record(
                    result.queue,
                    result.provenance,
                    result.decisions,
                    revision=int(prior_history["revision"]) + 1,
                    parent_report_sha256=prior_report_sha256,
                    new_decision_ids=new_decision_ids,
                ),
                "inputs": {
                    **dict(report.get("inputs") or {}),
                    "raw_decision_rows": raw_decisions.height,
                    "raw_decisions_fingerprint": _frame_fingerprint(raw_decisions),
                    "existing_decision_rows": existing.height,
                    "existing_decisions_fingerprint": _frame_fingerprint(existing),
                },
                "counts": {
                    **dict(report.get("counts") or {}),
                    "imported_decision_rows": imported_count,
                    "idempotent_replay_rows": replay_count,
                },
            }
        )
        result = ReferenceReviewWorkflowResult(
            queue=result.queue,
            provenance=result.provenance,
            decision_import=raw_decisions,
            decisions=result.decisions,
            outcomes=result.outcomes,
            conflicts=result.conflicts,
            verified=result.verified,
            excluded=result.excluded,
            report=report,
            markdown=_review_markdown(
                report,
                title="Reference review decision import",
            ),
        )
    except Exception as exc:
        _log_event(
            "reference_review_import_failed",
            **event_context,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    _log_event(
        "reference_review_import_completed",
        **event_context,
        counts=result.report["counts"],
    )
    return result


def resolve_reference_review_statuses(
    queue: pl.DataFrame,
    decisions: pl.DataFrame,
    *,
    queue_provenance: pl.DataFrame,
    resolved_at: datetime | None = None,
) -> ReferenceReviewWorkflowResult:
    run_started_at = datetime.now(UTC)
    validate_reference_review_queue(queue)
    _validate_queue_integrity(queue, queue_provenance)
    validate_reference_review_decisions(decisions)
    _validate_decision_source_hashes(decisions)
    _validate_workflow_foreign_keys(queue, decisions)
    timestamp = _resolution_timestamp(queue, decisions, resolved_at)
    queue_fingerprint = _queue_immutable_fingerprint(queue, queue_provenance)
    decision_fingerprint = _frame_fingerprint(decisions)

    decisions_by_request: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in decisions.iter_rows(named=True):
        decisions_by_request[str(row["review_request_id"])].append(dict(row))

    projected_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []
    completed_dispositions: dict[str, str] = {}
    resolved_projection_by_request: dict[str, dict[str, object]] = {}
    for queue_row in queue.iter_rows(named=True):
        request_id = str(queue_row["review_request_id"])
        history = decisions_by_request.get(request_id, [])
        effective = _effective_reviewer_decisions(history)
        resolution = _resolve_request(queue_row, effective, history=history)
        projected = dict(queue_row)
        projected["review_status"] = resolution["review_status"]
        projected_rows.append(projected)

        decisive = resolution["decisive"]
        assert isinstance(decisive, list)
        signature = resolution["resolved_signature"]
        effective_ids = sorted(str(row["review_decision_id"]) for row in effective)
        effective_actors = sorted(str(row["verified_by"]) for row in effective)
        blockers = _support_blockers(queue_row, resolution)
        outcome_rows.append(
            {
                "schema_version": REFERENCE_REVIEW_OUTCOMES_SCHEMA_VERSION,
                "review_request_id": request_id,
                "reference_media_id": queue_row["reference_media_id"],
                "review_status": resolution["review_status"],
                "effective_decision_ids": effective_ids,
                "effective_reviewer_ids": effective_actors,
                "decisive_reviewer_count": len(decisive),
                "required_review_count": queue_row["required_review_count"],
                "resolved_verification_status": signature[0] if signature else None,
                "target_identity_verified": signature[1] if signature else None,
                "life_stage": signature[2] if signature else None,
                "visual_domain": signature[3] if signature else None,
                "view": signature[4] if signature else None,
                "second_review_required": resolution["review_status"]
                == "second_review_required",
                "support_eligible": not blockers,
                "blocker_reasons": blockers,
                "queue_fingerprint": queue_fingerprint,
                "decision_ledger_fingerprint": decision_fingerprint,
                "resolved_at": timestamp,
            }
        )
        if resolution["review_status"] == "completed" and signature is not None:
            completed_dispositions[request_id] = str(signature[0])
            resolved_projection = {
                "schema_version": REFERENCE_REVIEW_RESOLVED_MEDIA_SCHEMA_VERSION,
                **{
                    field: projected[field]
                    for field in reference_review_queue_schema()
                    if field != "schema_version"
                },
                "resolved_verification_status": signature[0],
                "target_identity_verified": signature[1],
                "resolved_life_stage": signature[2],
                "resolved_visual_domain": signature[3],
                "resolved_view": signature[4],
                "effective_decision_ids": effective_ids,
                "effective_reviewer_ids": effective_actors,
                "resolved_exclusion_reasons": sorted(
                    {
                        str(row["exclusion_reason"])
                        for row in decisive
                        if row["exclusion_reason"]
                    }
                ),
            }
            resolved_projection_by_request[request_id] = resolved_projection
        conflict_rows.extend(
            _conflict_history_rows(
                queue_row,
                history,
                current_resolution=resolution,
            )
        )

    projected_queue = reference_review_queue_frame(projected_rows)
    projected_provenance = _project_queue_provenance(
        queue_provenance,
        projected_queue,
    )
    outcomes = _strict_frame(
        outcome_rows,
        schema=reference_review_outcome_schema(),
        sort_by=["reference_media_id", "review_request_id"],
    )
    conflicts = _strict_frame(
        conflict_rows,
        schema=reference_review_conflict_schema(),
        sort_by=["reference_media_id", "conflict_group_id"],
    )
    verified_ids = {
        request_id
        for request_id, status in completed_dispositions.items()
        if status == "verified"
    }
    excluded_ids = {
        request_id
        for request_id, status in completed_dispositions.items()
        if status == "excluded"
    }
    verified = _strict_frame(
        [resolved_projection_by_request[request_id] for request_id in verified_ids],
        schema=reference_review_resolved_media_schema(),
        sort_by=["reference_media_id", "review_request_id"],
    )
    excluded = _strict_frame(
        [resolved_projection_by_request[request_id] for request_id in excluded_ids],
        schema=reference_review_resolved_media_schema(),
        sort_by=["reference_media_id", "review_request_id"],
    )
    status_counts = Counter(str(value) for value in projected_queue["review_status"])
    run_ended_at = datetime.now(UTC)
    report = {
        "schema_version": REFERENCE_REVIEW_IMPORT_REPORT_VERSION,
        "command": "references.resolve_review_statuses",
        "status": "complete",
        "run_id": "not_instrumented",
        "pid": os.getpid(),
        "git_sha": current_git_sha(),
        "started_at": run_started_at.isoformat(),
        "ended_at": run_ended_at.isoformat(),
        "elapsed_seconds": max(
            0.0,
            (run_ended_at - run_started_at).total_seconds(),
        ),
        "resolved_at": timestamp.isoformat(),
        "inputs": {
            "queue_rows": queue.height,
            "queue_provenance_rows": queue_provenance.height,
            "decision_rows": decisions.height,
            "queue_fingerprint": queue_fingerprint,
            "source_queue_provenance_fingerprint": _frame_fingerprint(queue_provenance),
            "queue_provenance_fingerprint": _frame_fingerprint(projected_provenance),
            "decisions_fingerprint": decision_fingerprint,
        },
        "counts": {
            "queue_rows": projected_queue.height,
            "queue_provenance_rows": queue_provenance.height,
            "decision_rows": decisions.height,
            "outcome_rows": outcomes.height,
            "conflict_rows": conflicts.height,
            "verified_rows": verified.height,
            "excluded_rows": excluded.height,
        },
        "review_status_counts": dict(sorted(status_counts.items())),
        "outputs": {
            "queue_fingerprint": _frame_fingerprint(projected_queue),
            "queue_provenance_fingerprint": _frame_fingerprint(projected_provenance),
            "decisions_fingerprint": decision_fingerprint,
            "outcomes_fingerprint": _frame_fingerprint(outcomes),
            "conflicts_fingerprint": _frame_fingerprint(conflicts),
            "verified_fingerprint": _frame_fingerprint(verified),
            "excluded_fingerprint": _frame_fingerprint(excluded),
            "artifact_uris": "not_instrumented",
        },
        "metrics": {
            "api_calls": None,
            "retries": None,
            "rows_per_second": None,
            "peak_rss_bytes": "not_instrumented",
        },
    }
    result = ReferenceReviewWorkflowResult(
        queue=projected_queue,
        provenance=projected_provenance,
        decision_import=_strict_frame(
            [],
            schema=reference_review_decision_import_schema(),
            sort_by=_DECISION_IMPORT_SORT,
        ),
        decisions=decisions,
        outcomes=outcomes,
        conflicts=conflicts,
        verified=verified,
        excluded=excluded,
        report=report,
        markdown=_review_markdown(report, title="Reference review workflow"),
    )
    _validate_workflow_result(result)
    return result


def select_verified_reference_media(
    queue: pl.DataFrame,
    decisions: pl.DataFrame,
    *,
    queue_provenance: pl.DataFrame,
) -> pl.DataFrame:
    """Return only resolved, verified media that pass per-item support gates.

    Bank-level quota, diversity, and split checks remain readiness concerns.
    """

    result = resolve_reference_review_statuses(
        queue,
        decisions,
        queue_provenance=queue_provenance,
    )
    eligible_request_ids = set(
        result.outcomes.filter(pl.col("support_eligible"))["review_request_id"]
    )
    return _strict_frame(
        [
            row
            for row in result.verified.iter_rows(named=True)
            if str(row["review_request_id"]) in eligible_request_ids
        ],
        schema=reference_review_resolved_media_schema(),
        sort_by=["reference_media_id", "review_request_id"],
    )


def write_reference_review_export(
    result: ReferenceReviewQueueResult,
    output_dir: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, Path]:
    _validate_reference_review_export_result(result)
    empty_ledger = reference_review_decisions_frame([])
    return _write_review_packet(
        output_dir,
        run_id=run_id,
        command="references.export_review_queue",
        report=result.report,
        parquet_frames={
            "queue": ("reference_review_queue.parquet", result.queue, "queue"),
            "queue_provenance": (
                REFERENCE_REVIEW_QUEUE_PROVENANCE_FILE,
                result.provenance,
                "generic",
            ),
            "decision_template": (
                REFERENCE_REVIEW_DECISION_TEMPLATE_FILE,
                result.decision_template,
                "generic",
            ),
            "decisions": (
                REFERENCE_REVIEW_DECISIONS_FILE,
                empty_ledger,
                "decisions",
            ),
        },
        report_file=REFERENCE_REVIEW_EXPORT_REPORT_FILE,
        summary_file=REFERENCE_REVIEW_EXPORT_SUMMARY_FILE,
        title="Reference review queue export",
    )


def _validate_reference_review_export_result(
    result: ReferenceReviewQueueResult,
) -> None:
    validate_reference_review_queue(result.queue)
    _validate_queue_integrity(result.queue, result.provenance)
    if (
        result.report.get("schema_version") != REFERENCE_REVIEW_EXPORT_REPORT_VERSION
        or result.report.get("command") != "references.export_review_queue"
        or result.report.get("status") != "complete"
    ):
        raise ValueError("review queue export report identity is inconsistent")
    _validate_import_frame(result.decision_template, allow_incomplete=True)
    expected_template = reference_review_decision_template(
        result.queue,
        result.provenance,
    )
    if not result.decision_template.equals(expected_template):
        raise ValueError("review decision template does not match the exported queue")
    counts = result.report.get("counts")
    outputs = result.report.get("outputs")
    if (
        not isinstance(counts, Mapping)
        or counts.get("queue_rows") != result.queue.height
        or counts.get("queue_provenance_rows") != result.provenance.height
    ):
        raise ValueError("review queue export report count is inconsistent")
    if not isinstance(outputs, Mapping) or outputs.get(
        "queue_fingerprint"
    ) != _frame_fingerprint(result.queue):
        raise ValueError("review queue export report fingerprint is inconsistent")
    if outputs.get("decision_template_fingerprint") != _frame_fingerprint(
        result.decision_template
    ):
        raise ValueError("review decision template report fingerprint is inconsistent")
    if outputs.get("queue_provenance_fingerprint") != _frame_fingerprint(
        result.provenance
    ):
        raise ValueError("review queue provenance report fingerprint is inconsistent")
    empty_ledger = reference_review_decisions_frame([])
    if result.report.get("history") != _review_history_record(
        result.queue,
        result.provenance,
        empty_ledger,
        revision=0,
        parent_report_sha256=None,
        new_decision_ids=(),
    ):
        raise ValueError("review queue export history binding is inconsistent")


def write_reference_review_import(
    result: ReferenceReviewWorkflowResult,
    output_dir: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, Path]:
    _validate_publishable_workflow_result(result)
    return _write_review_packet(
        output_dir,
        run_id=run_id,
        command="references.import_review_decisions",
        report=result.report,
        parquet_frames={
            "queue": ("reference_review_queue.parquet", result.queue, "queue"),
            "queue_provenance": (
                REFERENCE_REVIEW_QUEUE_PROVENANCE_FILE,
                result.provenance,
                "generic",
            ),
            "decisions": (
                REFERENCE_REVIEW_DECISIONS_FILE,
                result.decisions,
                "decisions",
            ),
            "decision_import": (
                REFERENCE_REVIEW_DECISION_IMPORT_FILE,
                result.decision_import,
                "generic",
            ),
            "outcomes": (REFERENCE_REVIEW_OUTCOMES_FILE, result.outcomes, "generic"),
            "conflicts": (
                REFERENCE_REVIEW_CONFLICTS_FILE,
                result.conflicts,
                "generic",
            ),
            "verified": (
                REFERENCE_REVIEW_VERIFIED_FILE,
                result.verified,
                "generic",
            ),
            "excluded": (
                REFERENCE_REVIEW_EXCLUDED_FILE,
                result.excluded,
                "generic",
            ),
        },
        report_file=REFERENCE_REVIEW_IMPORT_REPORT_FILE,
        summary_file=REFERENCE_REVIEW_IMPORT_SUMMARY_FILE,
        title="Reference review decision import",
    )


def initialize_reference_review_history_head(
    history_head: str | Path,
    report_path: str | Path,
) -> None:
    report, report_sha256 = _load_published_review_report(report_path)
    _validate_published_review_packet(report, Path(report_path), report_sha256)
    history = report.get("history")
    if (
        report.get("command") != "references.export_review_queue"
        or not isinstance(history, Mapping)
        or history.get("revision") != 0
        or history.get("parent_report_sha256") is not None
    ):
        raise ValueError("review history must be initialized from a root export packet")
    state_path = Path(history_head)
    validate_reference_review_history_head_destination(
        state_path,
        Path(report_path).parent,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = _acquire_history_lock(state_path)
    try:
        if state_path.exists():
            raise FileExistsError(state_path)
        state = _history_head_record(
            history=history,
            report_path=Path(report_path),
            report_sha256=report_sha256,
        )
        _write_json_atomic_create(state_path, state)
    finally:
        _release_history_lock(lock_fd)


def validate_reference_review_history_head_destination(
    history_head: str | Path,
    packet_directory: str | Path,
) -> None:
    """Reject a root history destination that cannot safely own a packet."""

    state_path = Path(history_head)
    directory = Path(packet_directory)
    if state_path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(
            "review history head must be outside immutable packet directories"
        )
    if state_path.exists():
        raise FileExistsError(state_path)


def validate_reference_review_history_head(
    history_head: str | Path,
    report_path: str | Path,
) -> tuple[dict[str, object], str]:
    state_path = Path(history_head)
    lock_fd = _acquire_history_lock(state_path)
    try:
        state = _read_history_head(state_path)
        report, report_sha256 = _load_published_review_report(report_path)
        _validate_published_review_packet(report, Path(report_path), report_sha256)
        history = report.get("history")
        if not isinstance(history, Mapping):
            raise ValueError("review report has no history binding")
        expected = _history_head_record(
            history=history,
            report_path=Path(report_path),
            report_sha256=report_sha256,
        )
        if state != expected:
            raise ValueError("review packet is not the authoritative history head")
        return report, report_sha256
    finally:
        _release_history_lock(lock_fd)


def advance_reference_review_history_head(
    history_head: str | Path,
    *,
    prior_report_path: str | Path,
    next_report_path: str | Path,
) -> None:
    state_path = Path(history_head)
    lock_fd = _acquire_history_lock(state_path)
    try:
        state = _read_history_head(state_path)
        prior_report, prior_sha256 = _load_published_review_report(prior_report_path)
        next_report, next_sha256 = _load_published_review_report(next_report_path)
        _validate_published_review_packet(
            prior_report,
            Path(prior_report_path),
            prior_sha256,
        )
        _validate_published_review_packet(
            next_report,
            Path(next_report_path),
            next_sha256,
        )
        prior_history = prior_report.get("history")
        next_history = next_report.get("history")
        if not isinstance(prior_history, Mapping) or not isinstance(
            next_history, Mapping
        ):
            raise ValueError("review report has no history binding")
        expected_prior = _history_head_record(
            history=prior_history,
            report_path=Path(prior_report_path),
            report_sha256=prior_sha256,
        )
        if state != expected_prior:
            raise ValueError("review history head changed before publication")
        if (
            next_history.get("history_id") != prior_history.get("history_id")
            or next_history.get("revision") != int(prior_history["revision"]) + 1
            or next_history.get("parent_report_sha256") != prior_sha256
        ):
            raise ValueError("next review packet does not extend the history head")
        prior_decisions = _read_bound_review_parquet(prior_report, "decisions")
        next_decisions = _read_bound_review_parquet(next_report, "decisions")
        prior_provenance = _read_bound_review_parquet(
            prior_report,
            "queue_provenance",
        )
        prior_by_id = {
            str(row["review_decision_id"]): row
            for row in prior_decisions.iter_rows(named=True)
        }
        next_by_id = {
            str(row["review_decision_id"]): row
            for row in next_decisions.iter_rows(named=True)
        }
        if any(
            next_by_id.get(decision_id) != row
            for decision_id, row in prior_by_id.items()
        ):
            raise ValueError("next review packet rewrites prior decision history")
        expected_new_ids = sorted(set(next_by_id) - set(prior_by_id))
        if next_history.get("new_decision_ids") != expected_new_ids:
            raise ValueError("next review packet decision delta is inconsistent")
        decision_import = _read_bound_review_parquet(next_report, "decision_import")
        _validate_import_frame(decision_import)
        recomputed_next, imported_count, replay_count = _merge_decision_rows(
            prior_decisions,
            [
                _normalize_raw_decision(row)
                for row in decision_import.iter_rows(named=True)
            ],
        )
        if not recomputed_next.equals(next_decisions):
            raise ValueError("next review packet is not the imported ledger transition")
        next_inputs = next_report.get("inputs")
        next_counts = next_report.get("counts")
        if (
            not isinstance(next_inputs, Mapping)
            or next_inputs.get("existing_decision_rows") != prior_decisions.height
            or next_inputs.get("existing_decisions_fingerprint")
            != _frame_fingerprint(prior_decisions)
            or next_inputs.get("source_queue_provenance_fingerprint")
            != _frame_fingerprint(prior_provenance)
            or next_inputs.get("raw_decision_rows") != decision_import.height
            or next_inputs.get("raw_decisions_fingerprint")
            != _frame_fingerprint(decision_import)
            or not isinstance(next_counts, Mapping)
            or next_counts.get("imported_decision_rows") != imported_count
            or next_counts.get("idempotent_replay_rows") != replay_count
        ):
            raise ValueError("next review packet parent ledger audit is inconsistent")
        next_state = _history_head_record(
            history=next_history,
            report_path=Path(next_report_path),
            report_sha256=next_sha256,
        )
        _write_json_atomic_replace(state_path, next_state)
    finally:
        _release_history_lock(lock_fd)


def validate_reference_review_packet_artifact(
    report: Mapping[str, object],
    logical_name: str,
    artifact_path: str | Path,
) -> None:
    _validated_review_packet_artifact_content(report, logical_name, artifact_path)


def _validated_review_packet_artifact_content(
    report: Mapping[str, object],
    logical_name: str,
    artifact_path: str | Path,
) -> bytes:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("review report has no committed artifact ledger")
    record = artifacts.get(logical_name)
    if not isinstance(record, Mapping):
        raise ValueError(f"review report does not bind artifact: {logical_name}")
    path = Path(artifact_path)
    content = path.read_bytes()
    expected_uri = record.get("uri")
    if (
        record.get("committed") is not True
        or record.get("byte_count") != len(content)
        or record.get("sha256") != "sha256:" + hashlib.sha256(content).hexdigest()
        or not isinstance(expected_uri, str)
        or path.resolve() != Path(expected_uri).resolve()
    ):
        raise ValueError(f"review packet artifact binding is invalid: {logical_name}")
    return content


def _review_packet_artifact_path(
    report: Mapping[str, object],
    logical_name: str,
) -> Path:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("review report has no committed artifact ledger")
    record = artifacts.get(logical_name)
    if not isinstance(record, Mapping) or not isinstance(record.get("uri"), str):
        raise ValueError(f"review report does not bind artifact: {logical_name}")
    return Path(str(record["uri"]))


def _read_bound_review_parquet(
    report: Mapping[str, object],
    logical_name: str,
) -> pl.DataFrame:
    path = _review_packet_artifact_path(report, logical_name)
    content = _validated_review_packet_artifact_content(report, logical_name, path)
    return pl.read_parquet(io.BytesIO(content))


def _validate_published_review_packet(
    report: Mapping[str, object],
    report_path: Path,
    report_sha256: str,
) -> None:
    command = report.get("command")
    if command == "references.export_review_queue":
        expected_schema_version = REFERENCE_REVIEW_EXPORT_REPORT_VERSION
        expected_report_file = REFERENCE_REVIEW_EXPORT_REPORT_FILE
        expected_artifacts = {
            "queue": "reference_review_queue.parquet",
            "queue_provenance": REFERENCE_REVIEW_QUEUE_PROVENANCE_FILE,
            "decision_template": REFERENCE_REVIEW_DECISION_TEMPLATE_FILE,
            "decisions": REFERENCE_REVIEW_DECISIONS_FILE,
            "summary": REFERENCE_REVIEW_EXPORT_SUMMARY_FILE,
        }
    elif command == "references.import_review_decisions":
        expected_schema_version = REFERENCE_REVIEW_IMPORT_REPORT_VERSION
        expected_report_file = REFERENCE_REVIEW_IMPORT_REPORT_FILE
        expected_artifacts = {
            "queue": "reference_review_queue.parquet",
            "queue_provenance": REFERENCE_REVIEW_QUEUE_PROVENANCE_FILE,
            "decision_import": REFERENCE_REVIEW_DECISION_IMPORT_FILE,
            "decisions": REFERENCE_REVIEW_DECISIONS_FILE,
            "outcomes": REFERENCE_REVIEW_OUTCOMES_FILE,
            "conflicts": REFERENCE_REVIEW_CONFLICTS_FILE,
            "verified": REFERENCE_REVIEW_VERIFIED_FILE,
            "excluded": REFERENCE_REVIEW_EXCLUDED_FILE,
            "summary": REFERENCE_REVIEW_IMPORT_SUMMARY_FILE,
        }
    else:
        raise ValueError("review report command is incompatible")
    if (
        report.get("schema_version") != expected_schema_version
        or report.get("status") != "complete"
        or report_path.resolve().name != expected_report_file
        or _SHA256_PATTERN.fullmatch(report_sha256) is None
    ):
        raise ValueError("review report identity is incompatible")
    history = report.get("history")
    if not isinstance(history, Mapping):
        raise ValueError("review report has no history binding")
    _validate_review_history_record(history)
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_artifacts):
        raise ValueError("review report artifact ledger is incomplete")
    packet_directory = report_path.resolve().parent
    artifact_paths: dict[str, Path] = {}
    artifact_contents: dict[str, bytes] = {}
    for logical_name in sorted(expected_artifacts):
        record = artifacts[logical_name]
        if not isinstance(record, Mapping) or not isinstance(record.get("uri"), str):
            raise ValueError(f"review report artifact is malformed: {logical_name}")
        artifact_path = Path(str(record["uri"]))
        if (
            not artifact_path.is_absolute()
            or artifact_path.resolve().parent != packet_directory
            or artifact_path.name != expected_artifacts[logical_name]
        ):
            raise ValueError(f"review report artifact path is invalid: {logical_name}")
        artifact_contents[logical_name] = _validated_review_packet_artifact_content(
            report,
            logical_name,
            artifact_path,
        )
        artifact_paths[logical_name] = artifact_path
    outputs = report.get("outputs")
    expected_uris = {
        name: str(path.resolve()) for name, path in sorted(artifact_paths.items())
    }
    if (
        not isinstance(outputs, Mapping)
        or outputs.get("artifact_uris") != expected_uris
    ):
        raise ValueError("review report artifact URI projection is inconsistent")
    summary_report = json.loads(json.dumps(report))
    del summary_report["artifacts"]["summary"]
    del summary_report["outputs"]["artifact_uris"]["summary"]
    summary_title = (
        "Reference review queue export"
        if command == "references.export_review_queue"
        else "Reference review decision import"
    )
    expected_summary = _review_markdown(summary_report, title=summary_title).encode()
    if artifact_contents["summary"] != expected_summary:
        raise ValueError("review packet summary is not derived from its report")

    queue = pl.read_parquet(io.BytesIO(artifact_contents["queue"]))
    provenance = pl.read_parquet(io.BytesIO(artifact_contents["queue_provenance"]))
    decisions = pl.read_parquet(io.BytesIO(artifact_contents["decisions"]))
    if command == "references.export_review_queue":
        if not decisions.is_empty():
            raise ValueError("root review packet decision ledger must be empty")
        _validate_reference_review_export_result(
            ReferenceReviewQueueResult(
                queue=queue,
                provenance=provenance,
                decision_template=pl.read_parquet(
                    io.BytesIO(artifact_contents["decision_template"])
                ),
                report=dict(report),
                markdown="",
            )
        )
        return
    _validate_publishable_workflow_result(
        ReferenceReviewWorkflowResult(
            queue=queue,
            provenance=provenance,
            decision_import=pl.read_parquet(
                io.BytesIO(artifact_contents["decision_import"])
            ),
            decisions=decisions,
            outcomes=pl.read_parquet(io.BytesIO(artifact_contents["outcomes"])),
            conflicts=pl.read_parquet(io.BytesIO(artifact_contents["conflicts"])),
            verified=pl.read_parquet(io.BytesIO(artifact_contents["verified"])),
            excluded=pl.read_parquet(io.BytesIO(artifact_contents["excluded"])),
            report=dict(report),
            markdown="",
        )
    )


def _write_review_packet(
    output_dir: str | Path,
    *,
    run_id: str | None,
    command: str,
    report: Mapping[str, object],
    parquet_frames: Mapping[str, tuple[str, pl.DataFrame, str]],
    report_file: str,
    summary_file: str,
    title: str,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.{uuid4().hex}.tmp"
    started_at = datetime.now(UTC)
    effective_run_id = _required_canonical_text(
        run_id
        or (
            "reference-review-"
            + started_at.strftime("%Y%m%dT%H%M%S%fZ-")
            + uuid4().hex[:12]
        ),
        field="run_id",
    )
    lock_path = directory.parent / f".{directory.name}.publish.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(lock_fd)
        if isinstance(exc, BlockingIOError):
            raise FileExistsError(directory) from exc
        raise
    try:
        if directory.exists():
            raise FileExistsError(directory)
        _log_event(
            "reference_review_publication_started",
            command=command,
            run_id=effective_run_id,
            output_dir=str(directory),
            started_at=started_at.isoformat(),
        )
        staging.mkdir(parents=False, exist_ok=False)
        paths: dict[str, Path] = {}
        for key, (filename, frame, writer_kind) in parquet_frames.items():
            if writer_kind == "queue":
                path = write_reference_review_queue(frame, staging)
            elif writer_kind == "decisions":
                path = write_reference_review_decisions(frame, staging)
            else:
                path = write_parquet(frame, staging / filename, overwrite=False)
            paths[key] = path
        ended_at = datetime.now(UTC)
        artifact_records = {
            key: _local_artifact_record(path, final_directory=directory)
            for key, path in sorted(paths.items())
        }
        publication_report = json.loads(json.dumps(report))
        publication_report.update(
            {
                "command": command,
                "run_id": effective_run_id,
                "status": "complete",
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "elapsed_seconds": max(
                    0.0,
                    (ended_at - started_at).total_seconds(),
                ),
                "artifacts": artifact_records,
            }
        )
        outputs = dict(publication_report.get("outputs") or {})
        outputs["artifact_uris"] = {
            key: record["uri"] for key, record in artifact_records.items()
        }
        publication_report["outputs"] = outputs
        summary_path = staging / summary_file
        summary_path.write_text(
            _review_markdown(publication_report, title=title),
            encoding="utf-8",
        )
        paths["summary"] = summary_path
        publication_report["artifacts"]["summary"] = _local_artifact_record(
            summary_path,
            final_directory=directory,
        )
        publication_report["outputs"]["artifact_uris"]["summary"] = publication_report[
            "artifacts"
        ]["summary"]["uri"]
        report_path = staging / report_file
        report_path.write_text(
            json.dumps(publication_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths["report"] = report_path
        _rename_directory_no_replace(staging, directory)
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        ended_at = datetime.now(UTC)
        _log_event(
            "reference_review_publication_failed",
            command=command,
            run_id=effective_run_id,
            output_dir=str(directory),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        _write_failed_publication_audit(
            directory,
            command=command,
            run_id=effective_run_id,
            started_at=started_at,
            ended_at=ended_at,
            error=exc,
        )
        raise
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    final_paths = {
        key: (directory / path.name).resolve() for key, path in paths.items()
    }
    _log_event(
        "reference_review_publication_completed",
        command=command,
        run_id=effective_run_id,
        output_dir=str(directory),
        ended_at=ended_at.isoformat(),
        artifacts={key: str(path) for key, path in sorted(final_paths.items())},
    )
    return final_paths


def _write_failed_publication_audit(
    directory: Path,
    *,
    command: str,
    run_id: str,
    started_at: datetime,
    ended_at: datetime,
    error: Exception,
) -> None:
    failure_id = uuid4().hex
    stem = f".{directory.name}.{failure_id}.failed"
    report = {
        "schema_version": (
            REFERENCE_REVIEW_EXPORT_REPORT_VERSION
            if command == "references.export_review_queue"
            else REFERENCE_REVIEW_IMPORT_REPORT_VERSION
        ),
        "command": command,
        "run_id": run_id,
        "pid": os.getpid(),
        "git_sha": current_git_sha(),
        "status": "failed",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": max(0.0, (ended_at - started_at).total_seconds()),
        "output_dir": str(directory),
        "error_type": type(error).__name__,
        "error": str(error),
        "counts": {},
        "outputs": {"artifact_uris": "not_committed"},
    }
    try:
        json_path = directory.parent / f"{stem}.json"
        markdown_path = directory.parent / f"{stem}.md"
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(
            _review_markdown(report, title="Failed reference review publication"),
            encoding="utf-8",
        )
    except OSError:
        _LOGGER.exception(
            "could not persist failed reference review publication audit",
            extra={"output_dir": str(directory), "run_id": run_id},
        )


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        if destination.exists():
            raise FileExistsError(destination)
        source.rename(destination)
        return
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _load_published_review_report(
    report_path: str | Path,
) -> tuple[dict[str, object], str]:
    path = Path(report_path)
    content = path.read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("review report must contain a JSON object")
    if payload.get("status") != "complete":
        raise ValueError("review report is not complete")
    return payload, "sha256:" + hashlib.sha256(content).hexdigest()


def _history_head_record(
    *,
    history: Mapping[str, object],
    report_path: Path,
    report_sha256: str,
) -> dict[str, object]:
    _validate_review_history_record(history)
    if _SHA256_PATTERN.fullmatch(report_sha256) is None:
        raise ValueError("review report digest is invalid")
    return {
        "schema_version": REFERENCE_REVIEW_HISTORY_HEAD_SCHEMA_VERSION,
        "history_id": history["history_id"],
        "revision": history["revision"],
        "head_report_sha256": report_sha256,
        "head_report_path": str(report_path.resolve()),
    }


def _validate_review_history_record(history: Mapping[str, object]) -> None:
    required_fields = {
        "schema_version",
        "history_id",
        "revision",
        "parent_report_sha256",
        "queue_fingerprint",
        "queue_provenance_fingerprint",
        "decision_ledger_fingerprint",
        "new_decision_ids",
    }
    if set(history) != required_fields:
        raise ValueError("review history binding is malformed")
    revision = history.get("revision")
    parent = history.get("parent_report_sha256")
    new_ids = history.get("new_decision_ids")
    fingerprint_fields = (
        "queue_fingerprint",
        "queue_provenance_fingerprint",
        "decision_ledger_fingerprint",
    )
    if (
        history.get("schema_version") != REFERENCE_REVIEW_HISTORY_SCHEMA_VERSION
        or not isinstance(history.get("history_id"), str)
        or _HISTORY_ID_PATTERN.fullmatch(str(history["history_id"])) is None
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
        or (revision == 0 and parent is not None)
        or (
            revision > 0
            and (
                not isinstance(parent, str) or _SHA256_PATTERN.fullmatch(parent) is None
            )
        )
        or any(
            not isinstance(history.get(field), str)
            or _SHA256_PATTERN.fullmatch(str(history[field])) is None
            for field in fingerprint_fields
        )
        or not isinstance(new_ids, list)
        or any(not isinstance(value, str) or not value for value in new_ids)
        or new_ids != sorted(set(new_ids))
    ):
        raise ValueError("review history binding is incompatible")


def _history_lock_path(state_path: Path) -> Path:
    return state_path.with_name(f".{state_path.name}.lock")


def _acquire_history_lock(state_path: Path) -> int:
    lock_path = _history_lock_path(state_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    except OSError:
        os.close(lock_fd)
        raise
    return lock_fd


def _release_history_lock(lock_fd: int) -> None:
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _read_history_head(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "history_id",
        "revision",
        "head_report_sha256",
        "head_report_path",
    }:
        raise ValueError("reference review history head is malformed")
    if (
        payload["schema_version"] != REFERENCE_REVIEW_HISTORY_HEAD_SCHEMA_VERSION
        or not isinstance(payload["revision"], int)
        or isinstance(payload["revision"], bool)
        or payload["revision"] < 0
        or not isinstance(payload["head_report_sha256"], str)
        or _SHA256_PATTERN.fullmatch(payload["head_report_sha256"]) is None
    ):
        raise ValueError("reference review history head is incompatible")
    return payload


def _write_json_atomic_create(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(path) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic_replace(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _queue_row(
    queue_media_id: str,
    contexts: Sequence[Mapping[str, object]],
    *,
    reference_bank_version: str,
    created_at: datetime,
) -> tuple[dict[str, object], dict[str, object]]:
    first = contexts[0]
    obj = first["queue_object"]
    candidate = first["queue_candidate"]
    observation = first["queue_observation"]
    assert isinstance(obj, Mapping)
    assert isinstance(candidate, Mapping)
    assert isinstance(observation, Mapping)
    selection_rows = [context["selection"] for context in contexts]
    assert all(isinstance(row, Mapping) for row in selection_rows)
    taxonomy = sorted(
        {
            (
                str(row["candidate_accepted_taxon_key"]),
                str(row["scientific_name"]),
            )
            for row in selection_rows
        }
    )
    accepted_key: str | None
    scientific_name: str | None
    if len(taxonomy) == 1:
        accepted_key, scientific_name = taxonomy[0]
    else:
        accepted_key = None
        scientific_name = None

    unresolved_duplicate = any(
        bool(context["unresolved_duplicate"]) for context in contexts
    )
    reasons = {"manual_identity_review"}
    required_review_count = 1
    priority = 100
    if len(taxonomy) != 1:
        reasons.add("upstream_taxon_conflict")
        required_review_count = 2
        priority = min(priority, 10)
    if unresolved_duplicate:
        reasons.add("duplicate_resolution_pending")
        required_review_count = 2
        priority = min(priority, 10)
    if str(candidate["verification_status"]) == "needs_review":
        reasons.add("provider_review_requested")
        priority = min(priority, 20)
    if str(obj["licence_policy_status"]) == "research_only":
        reasons.add("research_only_licence")
        priority = min(priority, 30)

    life_stage = _single_proposal(
        (row["life_stage"] for row in selection_rows),
        choices=REFERENCE_LIFE_STAGES,
    )
    visual_domain = _single_proposal(
        (row["visual_domain"] for row in selection_rows),
        choices=REFERENCE_VISUAL_DOMAINS,
    )
    object_fingerprint = str(obj["object_fingerprint"])
    row = {
        "schema_version": REFERENCE_REVIEW_QUEUE_SCHEMA_VERSION,
        "reference_media_id": queue_media_id,
        "reference_observation_id": candidate["reference_observation_id"],
        "canonical_reference_media_id": obj["canonical_reference_media_id"],
        "accepted_taxon_key": accepted_key,
        "scientific_name": scientific_name,
        "durable_preview_uri": obj["source_object_uri"],
        "media_object_fingerprint": object_fingerprint,
        "duplicate_group_id": obj["duplicate_group_id"],
        "source": candidate["source"],
        "provider_media_id": candidate["provider_media_id"],
        "provider_verification_status": candidate["verification_status"],
        "creator": candidate["creator"],
        "rights_holder": candidate["rights_holder"],
        "licence": candidate["licence"],
        "licence_uri": candidate["licence_uri"],
        "licence_policy_status": obj["licence_policy_status"],
        "attribution": candidate["attribution"],
        "life_stage": life_stage,
        "visual_domain": visual_domain,
        "view": None,
        "review_reason": ";".join(sorted(reasons)),
        "review_priority": priority,
        "required_review_count": required_review_count,
        "review_status": "pending",
        "created_at": created_at,
        "reference_bank_version": reference_bank_version,
    }
    source_leaf_fingerprints = _queue_source_leaf_fingerprints(contexts)
    source_binding_fingerprint = _source_binding_fingerprint(source_leaf_fingerprints)
    queue_semantics_fingerprint = _queue_semantics_fingerprint(row)
    input_fingerprint = _review_input_fingerprint(
        queue_semantics_fingerprint=queue_semantics_fingerprint,
        source_binding_fingerprint=source_binding_fingerprint,
    )
    request_id = make_reference_review_request_id(
        reference_media_id=queue_media_id,
        media_object_fingerprint=object_fingerprint,
        reference_bank_version=reference_bank_version,
        input_fingerprint=input_fingerprint,
    )
    queue_row = {
        **row,
        "review_request_id": request_id,
        "input_fingerprint": input_fingerprint,
    }
    provenance_row = {
        "schema_version": REFERENCE_REVIEW_QUEUE_PROVENANCE_SCHEMA_VERSION,
        "review_request_id": request_id,
        "reference_media_id": queue_media_id,
        "source_binding_fingerprint": source_binding_fingerprint,
        "source_leaf_fingerprints": source_leaf_fingerprints,
        "queue_semantics_fingerprint": queue_semantics_fingerprint,
        "queue_row_fingerprint": _queue_row_fingerprint(queue_row),
        "input_fingerprint": input_fingerprint,
    }
    return queue_row, provenance_row


def _normalize_raw_decision(row: Mapping[str, object]) -> dict[str, object]:
    version = _required_canonical_text(
        row["import_schema_version"],
        field="import_schema_version",
    )
    if version != REFERENCE_REVIEW_DECISION_IMPORT_SCHEMA_VERSION:
        raise ValueError(f"unsupported review decision import schema: {version}")
    request_id = _required_canonical_text(
        row["review_request_id"],
        field="review_request_id",
    )
    media_id = _required_canonical_text(
        row["reference_media_id"],
        field="reference_media_id",
    )
    review_round = _positive_int(row["review_round"], field="review_round")
    actor = _canonical_reviewer_id(row["verified_by"])
    reviewed_at = _utc_datetime(row["reviewed_at"], field="reviewed_at")
    status = _choice(
        row["verification_status"],
        field="verification_status",
        choices=_DECISION_STATUSES,
    )
    identity = row["target_identity_verified"]
    if identity is not None and not isinstance(identity, bool):
        raise ValueError("target_identity_verified must be Boolean or null")
    life_stage = _choice(
        row["life_stage"],
        field="life_stage",
        choices=REFERENCE_LIFE_STAGES,
    )
    visual_domain = _choice(
        row["visual_domain"],
        field="visual_domain",
        choices=REFERENCE_VISUAL_DOMAINS,
    )
    view = _choice(row["view"], field="view", choices=REFERENCE_VIEWS)
    confidence = _choice(
        row["review_confidence"],
        field="review_confidence",
        choices=_REVIEW_CONFIDENCE_VALUES,
    )
    notes = _optional_canonical_text(row["review_notes"], field="review_notes")
    exclusion_reason = _optional_canonical_text(
        row["exclusion_reason"],
        field="exclusion_reason",
    )
    conflict_id = _optional_canonical_text(
        row["conflicts_with_decision_id"],
        field="conflicts_with_decision_id",
    )
    second_review_required = status == "uncertain"
    canonical_source = {
        "import_schema_version": version,
        "review_request_id": request_id,
        "reference_media_id": media_id,
        "review_round": review_round,
        "verified_by": actor,
        "reviewed_at": _datetime_text(reviewed_at),
        "target_identity_verified": identity,
        "verification_status": status,
        "life_stage": life_stage,
        "visual_domain": visual_domain,
        "view": view,
        "review_confidence": confidence,
        "review_notes": notes,
        "exclusion_reason": exclusion_reason,
        "conflicts_with_decision_id": conflict_id,
    }
    source_hash = _payload_hash(canonical_source)
    decision_id = make_reference_review_decision_id(
        review_request_id=request_id,
        reference_media_id=media_id,
        review_round=review_round,
        verified_by=actor,
        reviewed_at=reviewed_at,
        target_identity_verified=identity,
        verification_status=status,
        life_stage=life_stage,
        visual_domain=visual_domain,
        view=view,
        review_confidence=confidence,
        review_notes=notes,
        exclusion_reason=exclusion_reason,
        second_review_required=second_review_required,
        conflicts_with_decision_id=conflict_id,
    )
    return {
        "schema_version": REFERENCE_REVIEW_DECISIONS_SCHEMA_VERSION,
        "review_decision_id": decision_id,
        "review_request_id": request_id,
        "reference_media_id": media_id,
        "review_round": review_round,
        "verified_by": actor,
        "reviewed_at": reviewed_at,
        "target_identity_verified": identity,
        "verification_status": status,
        "life_stage": life_stage,
        "visual_domain": visual_domain,
        "view": view,
        "review_confidence": confidence,
        "review_notes": notes,
        "exclusion_reason": exclusion_reason,
        "second_review_required": second_review_required,
        "conflicts_with_decision_id": conflict_id,
        "decision_source_hash": source_hash,
    }


def _merge_decision_rows(
    existing: pl.DataFrame,
    imported_rows: Sequence[Mapping[str, object]],
) -> tuple[pl.DataFrame, int, int]:
    by_id = {
        str(row["review_decision_id"]): dict(row)
        for row in existing.iter_rows(named=True)
    }
    by_hash = {
        str(row["decision_source_hash"]): str(row["review_decision_id"])
        for row in existing.iter_rows(named=True)
    }
    imported_count = 0
    replay_count = 0
    for imported in imported_rows:
        decision_id = str(imported["review_decision_id"])
        source_hash = str(imported["decision_source_hash"])
        existing_row = by_id.get(decision_id)
        if existing_row is not None:
            if str(existing_row["decision_source_hash"]) != source_hash:
                raise ValueError(
                    "review decision ID collision has a different source hash"
                )
            replay_count += 1
            continue
        existing_id = by_hash.get(source_hash)
        if existing_id is not None and existing_id != decision_id:
            raise ValueError(
                "review decision source hash identifies different semantic decisions"
            )
        by_id[decision_id] = dict(imported)
        by_hash[source_hash] = decision_id
        imported_count += 1
    merged = reference_review_decisions_frame(list(by_id.values()))
    return merged, imported_count, replay_count


def _normalized_decision_source_hash(row: Mapping[str, object]) -> str:
    return _payload_hash(
        {
            "import_schema_version": REFERENCE_REVIEW_DECISION_IMPORT_SCHEMA_VERSION,
            "review_request_id": row["review_request_id"],
            "reference_media_id": row["reference_media_id"],
            "review_round": row["review_round"],
            "verified_by": row["verified_by"],
            "reviewed_at": _datetime_text(row["reviewed_at"]),
            "target_identity_verified": row["target_identity_verified"],
            "verification_status": row["verification_status"],
            "life_stage": row["life_stage"],
            "visual_domain": row["visual_domain"],
            "view": row["view"],
            "review_confidence": row["review_confidence"],
            "review_notes": row["review_notes"],
            "exclusion_reason": row["exclusion_reason"],
            "conflicts_with_decision_id": row["conflicts_with_decision_id"],
        }
    )


def _validate_decision_source_hashes(decisions: pl.DataFrame) -> None:
    for row in decisions.iter_rows(named=True):
        if row["decision_source_hash"] != _normalized_decision_source_hash(row):
            raise ValueError("decision source hash does not match canonical content")


def _validate_workflow_foreign_keys(
    queue: pl.DataFrame,
    decisions: pl.DataFrame,
) -> None:
    queue_by_request = {
        str(row["review_request_id"]): row for row in queue.iter_rows(named=True)
    }
    histories: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for decision in decisions.iter_rows(named=True):
        request_id = str(decision["review_request_id"])
        queue_row = queue_by_request.get(request_id)
        if queue_row is None:
            raise ValueError(f"decision references unknown review request {request_id}")
        if decision["reference_media_id"] != queue_row["reference_media_id"]:
            raise ValueError("decision media does not match its review request")
        _canonical_reviewer_id(decision["verified_by"])
        if decision["reviewed_at"] < queue_row["created_at"]:
            raise ValueError("review decision predates its immutable review request")
        histories[(request_id, str(decision["verified_by"]))].append(dict(decision))
    for (request_id, actor), rows in histories.items():
        rows.sort(key=lambda row: int(row["review_round"]))
        rounds = [int(row["review_round"]) for row in rows]
        if rounds != list(range(1, len(rounds) + 1)):
            raise ValueError(
                f"review rounds must be contiguous for {request_id} and {actor}"
            )
        timestamps = [row["reviewed_at"] for row in rows]
        if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError(
                f"review correction timestamps must increase for {request_id} and {actor}"
            )


def _validate_prior_review_projection(
    queue: pl.DataFrame,
    provenance: pl.DataFrame,
    decisions: pl.DataFrame,
) -> None:
    expected = resolve_reference_review_statuses(
        queue,
        decisions,
        queue_provenance=provenance,
    )
    if not queue.equals(expected.queue):
        raise ValueError(
            "review queue is not the projection of the complete existing decision ledger"
        )
    if not provenance.equals(expected.provenance):
        raise ValueError(
            "review queue provenance is not the projection of the complete existing "
            "decision ledger"
        )


def _effective_reviewer_decisions(
    decisions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for decision in decisions:
        actor = str(decision["verified_by"])
        current = latest.get(actor)
        if current is None or int(decision["review_round"]) > int(
            current["review_round"]
        ):
            latest[actor] = dict(decision)
    return [latest[actor] for actor in sorted(latest)]


def _conflict_history_rows(
    queue_row: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
    *,
    current_resolution: Mapping[str, object],
) -> list[dict[str, object]]:
    snapshots: dict[str, dict[str, object]] = {}
    partial_history: list[dict[str, object]] = []
    historical_queue_row = dict(queue_row)
    historical_queue_row["review_status"] = "pending"
    for decision in sorted(
        history,
        key=lambda row: (row["reviewed_at"], str(row["review_decision_id"])),
    ):
        partial_history.append(dict(decision))
        effective = _effective_reviewer_decisions(partial_history)
        resolution = _resolve_request(
            historical_queue_row,
            effective,
            history=partial_history,
        )
        if resolution["review_status"] != "conflict":
            continue
        decision_ids = sorted(str(row["review_decision_id"]) for row in effective)
        actor_ids = sorted(str(row["verified_by"]) for row in effective)
        fields = resolution["conflicting_fields"]
        assert isinstance(fields, list)
        group_id = _conflict_group_id(
            str(queue_row["review_request_id"]),
            decision_ids,
            fields,
        )
        snapshots.setdefault(
            group_id,
            {
                "schema_version": REFERENCE_REVIEW_CONFLICTS_SCHEMA_VERSION,
                "conflict_group_id": group_id,
                "review_request_id": queue_row["review_request_id"],
                "reference_media_id": queue_row["reference_media_id"],
                "effective_decision_ids": decision_ids,
                "effective_reviewer_ids": actor_ids,
                "conflicting_fields": fields,
                "resolution_status": "resolved",
                "detected_at": decision["reviewed_at"],
            },
        )
    if current_resolution["review_status"] == "conflict":
        current_ids = sorted(
            str(row["review_decision_id"])
            for row in _effective_reviewer_decisions(history)
        )
        current_fields = current_resolution["conflicting_fields"]
        assert isinstance(current_fields, list)
        current_group_id = _conflict_group_id(
            str(queue_row["review_request_id"]),
            current_ids,
            current_fields,
        )
        current_snapshot = snapshots.get(current_group_id)
        if current_snapshot is None:
            raise ValueError("current review conflict has no historical snapshot")
        current_snapshot["resolution_status"] = "open"
    return [snapshots[key] for key in sorted(snapshots)]


def _resolve_request(
    queue_row: Mapping[str, object],
    effective: Sequence[Mapping[str, object]],
    *,
    history: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if queue_row["review_status"] == "cancelled":
        raise ValueError(
            "cancelled review requests require an attributable cancellation record"
        )
    if not effective:
        return {
            "review_status": "pending",
            "decisive": [],
            "resolved_signature": None,
            "conflicting_fields": [],
        }
    decisive = [row for row in effective if row["verification_status"] != "uncertain"]
    fields = [
        "verification_status",
        "target_identity_verified",
        "life_stage",
        "visual_domain",
        "view",
    ]
    conflicting_fields = [
        field
        for field in fields
        if len({_json_scalar(row[field]) for row in decisive}) > 1
    ]
    if any(row["conflicts_with_decision_id"] is not None for row in effective):
        conflicting_fields.append("explicit_conflict_pointer")
    conflicting_fields = sorted(set(conflicting_fields))
    if conflicting_fields:
        return {
            "review_status": "conflict",
            "decisive": decisive,
            "resolved_signature": None,
            "conflicting_fields": conflicting_fields,
        }

    unresolved_uncertain = False
    uncertain_history = [
        row for row in history if row["verification_status"] == "uncertain"
    ]
    for abstention in uncertain_history:
        abstention_actor = str(abstention["verified_by"])
        abstention_time = abstention["reviewed_at"]
        if not any(
            str(row["verified_by"]) != abstention_actor
            and row["reviewed_at"] > abstention_time
            for row in decisive
        ):
            unresolved_uncertain = True
            break
    required = int(queue_row["required_review_count"])
    if unresolved_uncertain or len(decisive) < required:
        return {
            "review_status": "second_review_required",
            "decisive": decisive,
            "resolved_signature": None,
            "conflicting_fields": [],
        }
    signature = _decision_signature(decisive[0]) if decisive else None
    return {
        "review_status": "completed",
        "decisive": decisive,
        "resolved_signature": signature,
        "conflicting_fields": [],
    }


def _decision_signature(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["verification_status"],
        row["target_identity_verified"],
        row["life_stage"],
        row["visual_domain"],
        row["view"],
    )


def _support_blockers(
    queue_row: Mapping[str, object],
    resolution: Mapping[str, object],
) -> list[str]:
    blockers: list[str] = []
    signature = resolution["resolved_signature"]
    if resolution["review_status"] != "completed" or signature is None:
        blockers.append("review_not_completed")
    else:
        if signature[0] != "verified" or signature[1] is not True:
            blockers.append("not_verified")
        if signature[3] not in _PRODUCTION_VISUAL_DOMAINS:
            blockers.append("visual_domain_not_support_eligible")
    if queue_row["reference_media_id"] != queue_row["canonical_reference_media_id"]:
        blockers.append("noncanonical_media")
    if queue_row["accepted_taxon_key"] is None or queue_row["scientific_name"] is None:
        blockers.append("taxonomy_unresolved")
    if queue_row["licence_policy_status"] != "allowed":
        blockers.append("licence_not_allowed")
    if not queue_row["attribution"]:
        blockers.append("attribution_missing")
    if "duplicate_resolution_pending" in str(queue_row["review_reason"]).split(";"):
        blockers.append("duplicate_resolution_pending")
    return sorted(set(blockers))


def _validate_workflow_result(result: ReferenceReviewWorkflowResult) -> None:
    validate_reference_review_queue(result.queue)
    _validate_queue_integrity(result.queue, result.provenance)
    _validate_import_frame(result.decision_import)
    validate_reference_review_decisions(result.decisions)
    _validate_workflow_foreign_keys(result.queue, result.decisions)
    _validate_exact_schema(
        result.outcomes, reference_review_outcome_schema(), "outcomes"
    )
    _validate_exact_schema(
        result.conflicts, reference_review_conflict_schema(), "conflicts"
    )
    _validate_exact_schema(
        result.verified,
        reference_review_resolved_media_schema(),
        "verified reference media",
    )
    _validate_exact_schema(
        result.excluded,
        reference_review_resolved_media_schema(),
        "excluded reference media",
    )
    if not set(result.verified["review_request_id"]).isdisjoint(
        set(result.excluded["review_request_id"])
    ):
        raise ValueError("a review request cannot be both verified and excluded")
    completed = result.queue.filter(pl.col("review_status") == "completed")
    projected_ids = set(result.verified["review_request_id"]) | set(
        result.excluded["review_request_id"]
    )
    if set(completed["review_request_id"]) != projected_ids:
        raise ValueError(
            "completed review queue rows must have one resolved disposition"
        )
    outcome_by_request = {
        str(row["review_request_id"]): row
        for row in result.outcomes.iter_rows(named=True)
    }
    if set(outcome_by_request) != set(result.queue["review_request_id"]):
        raise ValueError("review outcomes must cover the complete queue")
    queue_by_request = {
        str(row["review_request_id"]): row for row in result.queue.iter_rows(named=True)
    }
    for artifact, expected_disposition in (
        (result.verified, "verified"),
        (result.excluded, "excluded"),
    ):
        for row in artifact.iter_rows(named=True):
            request_id = str(row["review_request_id"])
            outcome = outcome_by_request[request_id]
            queue_row = queue_by_request[request_id]
            if any(
                row[field] != queue_row[field]
                for field in reference_review_queue_schema()
                if field != "schema_version"
            ):
                raise ValueError(
                    "resolved media projection changed its immutable review request"
                )
            if (
                row["schema_version"] != REFERENCE_REVIEW_RESOLVED_MEDIA_SCHEMA_VERSION
                or outcome["review_status"] != "completed"
                or outcome["resolved_verification_status"] != expected_disposition
                or row["resolved_verification_status"] != expected_disposition
                or row["target_identity_verified"]
                != outcome["target_identity_verified"]
                or row["effective_decision_ids"] != outcome["effective_decision_ids"]
                or row["effective_reviewer_ids"] != outcome["effective_reviewer_ids"]
            ):
                raise ValueError("resolved media projection is inconsistent")
            if (
                expected_disposition == "verified"
                and row["target_identity_verified"] is not True
            ):
                raise ValueError("verified projection contains an unresolved decision")
            for resolved_field, outcome_field in (
                ("resolved_life_stage", "life_stage"),
                ("resolved_visual_domain", "visual_domain"),
                ("resolved_view", "view"),
            ):
                if row[resolved_field] != outcome[outcome_field]:
                    raise ValueError(
                        "resolved media projection does not contain the human review"
                    )


def _validate_publishable_workflow_result(
    result: ReferenceReviewWorkflowResult,
) -> None:
    _validate_workflow_result(result)
    if (
        result.report.get("schema_version") != REFERENCE_REVIEW_IMPORT_REPORT_VERSION
        or result.report.get("command") != "references.import_review_decisions"
        or result.report.get("status") != "complete"
    ):
        raise ValueError("review workflow report identity is inconsistent")
    resolved_times = set(result.outcomes["resolved_at"].drop_nulls().to_list())
    if len(resolved_times) > 1:
        raise ValueError("review outcomes cannot contain multiple resolution times")
    resolved_at = next(iter(resolved_times), None)
    report_resolved_at = result.report.get("resolved_at")
    if resolved_at is not None:
        if report_resolved_at != resolved_at.isoformat():
            raise ValueError("review workflow resolution timestamp is inconsistent")
    elif not isinstance(report_resolved_at, str):
        raise ValueError("review workflow resolution timestamp is missing")
    else:
        try:
            parsed_report_resolved_at = datetime.fromisoformat(report_resolved_at)
        except ValueError as exc:
            raise ValueError("review workflow resolution timestamp is invalid") from exc
        if (
            parsed_report_resolved_at.tzinfo is None
            or parsed_report_resolved_at.utcoffset() != UTC.utcoffset(None)
            or report_resolved_at != parsed_report_resolved_at.isoformat()
        ):
            raise ValueError("review workflow resolution timestamp is invalid")
    expected = resolve_reference_review_statuses(
        result.queue,
        result.decisions,
        queue_provenance=result.provenance,
        resolved_at=resolved_at,
    )
    for field in (
        "queue",
        "provenance",
        "outcomes",
        "conflicts",
        "verified",
        "excluded",
    ):
        if not getattr(result, field).equals(getattr(expected, field)):
            raise ValueError(f"review workflow {field} is not the derived projection")
    expected_counts = {
        "queue_rows": result.queue.height,
        "queue_provenance_rows": result.provenance.height,
        "decision_rows": result.decisions.height,
        "outcome_rows": result.outcomes.height,
        "conflict_rows": result.conflicts.height,
        "verified_rows": result.verified.height,
        "excluded_rows": result.excluded.height,
    }
    report_counts = result.report.get("counts")
    if not isinstance(report_counts, Mapping):
        raise ValueError("review workflow report counts are missing")
    for key, value in expected_counts.items():
        if report_counts.get(key) != value:
            raise ValueError(f"review workflow report count is inconsistent: {key}")
    expected_fingerprints = {
        "queue_fingerprint": _frame_fingerprint(result.queue),
        "queue_provenance_fingerprint": _frame_fingerprint(result.provenance),
        "decisions_fingerprint": _frame_fingerprint(result.decisions),
        "outcomes_fingerprint": _frame_fingerprint(result.outcomes),
        "conflicts_fingerprint": _frame_fingerprint(result.conflicts),
        "verified_fingerprint": _frame_fingerprint(result.verified),
        "excluded_fingerprint": _frame_fingerprint(result.excluded),
    }
    outputs = result.report.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("review workflow report outputs are missing")
    for key, value in expected_fingerprints.items():
        if outputs.get(key) != value:
            raise ValueError(
                f"review workflow report fingerprint is inconsistent: {key}"
            )
    expected_status_counts = dict(
        sorted(Counter(str(value) for value in result.queue["review_status"]).items())
    )
    if result.report.get("review_status_counts") != expected_status_counts:
        raise ValueError("review workflow status counts are inconsistent")
    history = result.report.get("history")
    if not isinstance(history, Mapping):
        raise ValueError("review workflow history binding is missing")
    _validate_review_history_record(history)
    revision = history.get("revision")
    parent = history.get("parent_report_sha256")
    new_ids = history.get("new_decision_ids")
    expected_history = {
        "schema_version": REFERENCE_REVIEW_HISTORY_SCHEMA_VERSION,
        "history_id": _review_history_id(result.queue, result.provenance),
        "queue_fingerprint": _frame_fingerprint(result.queue),
        "queue_provenance_fingerprint": _frame_fingerprint(result.provenance),
        "decision_ledger_fingerprint": _frame_fingerprint(result.decisions),
    }
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
        or not isinstance(parent, str)
        or _SHA256_PATTERN.fullmatch(parent) is None
        or not isinstance(new_ids, list)
        or new_ids != sorted(set(new_ids))
        or not set(new_ids) <= set(result.decisions["review_decision_id"])
        or any(history.get(field) != value for field, value in expected_history.items())
    ):
        raise ValueError("review workflow history binding is inconsistent")
    assert isinstance(new_ids, list)
    normalized_import_rows = [
        _normalize_raw_decision(row)
        for row in result.decision_import.iter_rows(named=True)
    ]
    decision_by_id = {
        str(row["review_decision_id"]): row
        for row in result.decisions.iter_rows(named=True)
    }
    if not set(new_ids) <= {
        str(row["review_decision_id"]) for row in normalized_import_rows
    } or any(
        decision_by_id.get(str(row["review_decision_id"])) != row
        for row in normalized_import_rows
    ):
        raise ValueError("review decision import artifact is inconsistent")
    imported_count = len(new_ids)
    replay_count = result.decision_import.height - imported_count
    if replay_count < 0:
        raise ValueError("review decision import row counts are inconsistent")
    if (
        report_counts.get("imported_decision_rows") != imported_count
        or report_counts.get("idempotent_replay_rows") != replay_count
    ):
        raise ValueError("review workflow import counts are inconsistent")
    inputs = result.report.get("inputs")
    expected_inputs = {
        "queue_rows": result.queue.height,
        "queue_provenance_rows": result.provenance.height,
        "decision_rows": result.decisions.height,
        "queue_fingerprint": _queue_immutable_fingerprint(
            result.queue,
            result.provenance,
        ),
        "queue_provenance_fingerprint": _frame_fingerprint(result.provenance),
        "decisions_fingerprint": _frame_fingerprint(result.decisions),
        "raw_decision_rows": result.decision_import.height,
        "raw_decisions_fingerprint": _frame_fingerprint(result.decision_import),
        "existing_decision_rows": result.decisions.height - imported_count,
    }
    if not isinstance(inputs, Mapping) or any(
        inputs.get(key) != value for key, value in expected_inputs.items()
    ):
        raise ValueError("review workflow report inputs are inconsistent")
    existing_fingerprint = inputs.get("existing_decisions_fingerprint")
    source_provenance_fingerprint = inputs.get("source_queue_provenance_fingerprint")
    if (
        not isinstance(existing_fingerprint, str)
        or _SHA256_PATTERN.fullmatch(existing_fingerprint) is None
        or not isinstance(source_provenance_fingerprint, str)
        or _SHA256_PATTERN.fullmatch(source_provenance_fingerprint) is None
    ):
        raise ValueError("review workflow parent audit fingerprint is invalid")


def _queue_semantics_fingerprint(row: Mapping[str, object]) -> str:
    operational_or_generated = {
        "created_at",
        "durable_preview_uri",
        "review_priority",
        "review_reason",
        "review_request_id",
        "review_status",
        "input_fingerprint",
    }
    payload = {
        field: row[field]
        for field in reference_review_queue_schema()
        if field not in operational_or_generated and field in row
    }
    missing = sorted(
        set(reference_review_queue_schema()) - operational_or_generated - set(payload)
    )
    if missing:
        raise ValueError(f"review queue integrity fields are missing: {missing}")
    return _payload_hash(
        {
            "domain": "biominer.reference-review.queue-semantics.v1",
            "queue": payload,
        }
    )


def _review_input_fingerprint(
    *,
    queue_semantics_fingerprint: str,
    source_binding_fingerprint: str,
) -> str:
    return _payload_hash(
        {
            "domain": "biominer.reference-review.request-input.v1",
            "queue_semantics_fingerprint": queue_semantics_fingerprint,
            "source_binding_fingerprint": source_binding_fingerprint,
        }
    )


def _queue_row_fingerprint(row: Mapping[str, object]) -> str:
    payload = {field: row[field] for field in reference_review_queue_schema()}
    return _payload_hash(
        {"domain": "biominer.reference-review.queue-row.v1", "queue": payload}
    )


def _queue_source_leaf_fingerprints(
    contexts: Sequence[Mapping[str, object]],
) -> list[str]:
    leaves: dict[tuple[str, str, str], str] = {}

    def add(
        source_kind: str,
        source_role: str,
        source_record_id: object,
        source_row: object,
    ) -> None:
        if not isinstance(source_row, Mapping):
            raise ValueError("review source provenance row must be a mapping")
        key = (source_kind, source_role, str(source_record_id))
        fingerprint = _payload_hash(
            {
                "domain": "biominer.reference-review.source-leaf.v1",
                "source_kind": source_kind,
                "source_role": source_role,
                "source_record_id": str(source_record_id),
                "source_row": _source_binding_projection(source_kind, source_row),
            }
        )
        previous = leaves.setdefault(key, fingerprint)
        if previous != fingerprint:
            raise ValueError("one review source record has conflicting provenance")

    for context in contexts:
        selection = context["selection"]
        selected_object = context["selected_object"]
        selected_candidate = context["selected_candidate"]
        selected_observation = context["selected_observation"]
        queue_object = context["queue_object"]
        queue_candidate = context["queue_candidate"]
        queue_observation = context["queue_observation"]
        assert isinstance(selection, Mapping)
        assert isinstance(selected_object, Mapping)
        assert isinstance(selected_candidate, Mapping)
        assert isinstance(selected_observation, Mapping)
        assert isinstance(queue_object, Mapping)
        assert isinstance(queue_candidate, Mapping)
        assert isinstance(queue_observation, Mapping)
        add(
            "selection",
            "selected",
            selection["reference_selection_id"],
            selection,
        )
        add(
            "media_object",
            "selected",
            selected_object["reference_media_id"],
            selected_object,
        )
        add(
            "media_candidate",
            "selected",
            selected_candidate["reference_media_id"],
            selected_candidate,
        )
        add(
            "observation",
            "selected",
            selected_observation["reference_observation_id"],
            selected_observation,
        )
        add(
            "media_object",
            "canonical",
            queue_object["reference_media_id"],
            queue_object,
        )
        add(
            "media_candidate",
            "canonical",
            queue_candidate["reference_media_id"],
            queue_candidate,
        )
        add(
            "observation",
            "canonical",
            queue_observation["reference_observation_id"],
            queue_observation,
        )
        group_members = context["group_members"]
        assert isinstance(group_members, Sequence)
        for member in group_members:
            assert isinstance(member, Sequence) and len(member) == 3
            member_object, member_candidate, member_observation = member
            assert isinstance(member_object, Mapping)
            assert isinstance(member_candidate, Mapping)
            assert isinstance(member_observation, Mapping)
            add(
                "media_object",
                "group_evidence",
                member_object["reference_media_id"],
                member_object,
            )
            add(
                "media_candidate",
                "group_evidence",
                member_candidate["reference_media_id"],
                member_candidate,
            )
            add(
                "observation",
                "group_evidence",
                member_observation["reference_observation_id"],
                member_observation,
            )
        relationships = context["relationships"]
        assert isinstance(relationships, Sequence)
        for relationship in relationships:
            assert isinstance(relationship, Mapping)
            add(
                "duplicate_relationship",
                "group_evidence",
                relationship["duplicate_relationship_id"],
                relationship,
            )
    return sorted(leaves.values())


def _source_binding_projection(
    source_kind: str,
    source_row: Mapping[str, object],
) -> dict[str, object]:
    operational_fields = {
        "media_candidate": {
            "download_status",
            "media_identifier",
            "retrieved_at",
        },
        "media_object": {
            "download_attempt_count",
            "downloaded_at",
            "source_object_uri",
        },
        "observation": {"retrieved_at", "source_record_url"},
        "selection": {"selected_at"},
        "duplicate_relationship": set(),
    }
    if source_kind not in operational_fields:
        raise ValueError(f"unsupported review source kind: {source_kind}")
    excluded = operational_fields[source_kind]
    return {
        str(field): value
        for field, value in sorted(source_row.items())
        if field not in excluded
    }


def _source_binding_fingerprint(source_leaf_fingerprints: Sequence[str]) -> str:
    return _payload_hash(
        {
            "domain": "biominer.reference-review.source-set.v1",
            "source_leaf_fingerprints": sorted(source_leaf_fingerprints),
        }
    )


def _project_queue_provenance(
    provenance: pl.DataFrame,
    queue: pl.DataFrame,
) -> pl.DataFrame:
    queue_by_request = {
        str(row["review_request_id"]): row for row in queue.iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    for source in provenance.iter_rows(named=True):
        row = dict(source)
        queue_row = queue_by_request[str(row["review_request_id"])]
        row["queue_row_fingerprint"] = _queue_row_fingerprint(queue_row)
        rows.append(row)
    return _strict_frame(
        rows,
        schema=reference_review_queue_provenance_schema(),
        sort_by=["reference_media_id", "review_request_id"],
    )


def _review_history_id(queue: pl.DataFrame, provenance: pl.DataFrame) -> str:
    digest = _payload_hash(
        {
            "schema_version": REFERENCE_REVIEW_HISTORY_SCHEMA_VERSION,
            "root_queue_fingerprint": _queue_immutable_fingerprint(
                queue,
                provenance,
            ),
        }
    ).removeprefix("sha256:")
    return f"reference-review-history:{digest}"


def _review_history_record(
    queue: pl.DataFrame,
    provenance: pl.DataFrame,
    decisions: pl.DataFrame,
    *,
    revision: int,
    parent_report_sha256: str | None,
    new_decision_ids: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": REFERENCE_REVIEW_HISTORY_SCHEMA_VERSION,
        "history_id": _review_history_id(queue, provenance),
        "revision": revision,
        "parent_report_sha256": parent_report_sha256,
        "queue_fingerprint": _frame_fingerprint(queue),
        "queue_provenance_fingerprint": _frame_fingerprint(provenance),
        "decision_ledger_fingerprint": _frame_fingerprint(decisions),
        "new_decision_ids": sorted(new_decision_ids),
    }


def _validate_prior_review_report(
    queue: pl.DataFrame,
    provenance: pl.DataFrame,
    decisions: pl.DataFrame,
    *,
    prior_report: Mapping[str, object],
    prior_report_sha256: str,
) -> Mapping[str, object]:
    if _SHA256_PATTERN.fullmatch(prior_report_sha256) is None:
        raise ValueError("prior review report digest is invalid")
    if prior_report.get("status") != "complete" or prior_report.get("command") not in {
        "references.export_review_queue",
        "references.import_review_decisions",
    }:
        raise ValueError("prior review report is not a completed review packet")
    history = prior_report.get("history")
    if not isinstance(history, Mapping):
        raise ValueError("prior review report has no history binding")
    _validate_review_history_record(history)
    revision = history.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("prior review history revision is invalid")
    parent = history.get("parent_report_sha256")
    if revision == 0:
        if parent is not None or not decisions.is_empty():
            raise ValueError("root review history must have an empty decision ledger")
    elif not isinstance(parent, str) or _SHA256_PATTERN.fullmatch(parent) is None:
        raise ValueError("prior review history parent digest is invalid")
    expected = {
        "schema_version": REFERENCE_REVIEW_HISTORY_SCHEMA_VERSION,
        "history_id": _review_history_id(queue, provenance),
        "queue_fingerprint": _frame_fingerprint(queue),
        "queue_provenance_fingerprint": _frame_fingerprint(provenance),
        "decision_ledger_fingerprint": _frame_fingerprint(decisions),
    }
    if any(history.get(field) != value for field, value in expected.items()):
        raise ValueError("prior review report does not bind the supplied packet")
    new_ids = history.get("new_decision_ids")
    if not isinstance(new_ids, list) or new_ids != sorted(set(new_ids)):
        raise ValueError("prior review history decision delta is invalid")
    return history


def _queue_immutable_fingerprint(
    queue: pl.DataFrame,
    provenance: pl.DataFrame,
) -> str:
    provenance_by_request = {
        str(row["review_request_id"]): row for row in provenance.iter_rows(named=True)
    }
    return _payload_hash(
        [
            {
                "review_request_id": row["review_request_id"],
                "input_fingerprint": row["input_fingerprint"],
                "source_binding_fingerprint": provenance_by_request[
                    str(row["review_request_id"])
                ]["source_binding_fingerprint"],
            }
            for row in queue.iter_rows(named=True)
        ]
    )


def _validate_queue_integrity(
    queue: pl.DataFrame,
    provenance: pl.DataFrame,
) -> None:
    _validate_exact_schema(
        provenance,
        reference_review_queue_provenance_schema(),
        "reference review queue provenance",
    )
    if provenance["review_request_id"].n_unique() != provenance.height:
        raise ValueError("review queue provenance has duplicate review requests")
    if set(queue["review_request_id"]) != set(provenance["review_request_id"]):
        raise ValueError("review queue provenance must cover the complete queue")
    provenance_by_request = {
        str(row["review_request_id"]): row for row in provenance.iter_rows(named=True)
    }
    for row in queue.iter_rows(named=True):
        source = provenance_by_request[str(row["review_request_id"])]
        if source["schema_version"] != REFERENCE_REVIEW_QUEUE_PROVENANCE_SCHEMA_VERSION:
            raise ValueError(
                "review queue provenance has an unsupported schema version"
            )
        if source["reference_media_id"] != row["reference_media_id"]:
            raise ValueError("review queue provenance media binding is inconsistent")
        leaf_fingerprints = source["source_leaf_fingerprints"]
        if not leaf_fingerprints or leaf_fingerprints != sorted(set(leaf_fingerprints)):
            raise ValueError(
                "review queue provenance leaves must be nonempty, sorted, and unique"
            )
        fingerprint_values = [
            *leaf_fingerprints,
            source["source_binding_fingerprint"],
            source["queue_semantics_fingerprint"],
            source["queue_row_fingerprint"],
            source["input_fingerprint"],
        ]
        if any(
            not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None
            for value in fingerprint_values
        ):
            raise ValueError("review queue provenance has an invalid fingerprint")
        expected_source = _source_binding_fingerprint(leaf_fingerprints)
        if source["source_binding_fingerprint"] != expected_source:
            raise ValueError("review queue source binding is inconsistent")
        expected_semantics = _queue_semantics_fingerprint(row)
        if source["queue_semantics_fingerprint"] != expected_semantics:
            raise ValueError("review queue semantics binding is inconsistent")
        expected_input = _review_input_fingerprint(
            queue_semantics_fingerprint=expected_semantics,
            source_binding_fingerprint=expected_source,
        )
        if (
            source["input_fingerprint"] != expected_input
            or row["input_fingerprint"] != expected_input
        ):
            raise ValueError("review queue input fingerprint is inconsistent")
        if source["queue_row_fingerprint"] != _queue_row_fingerprint(row):
            raise ValueError("review queue row fingerprint is inconsistent")


def _validate_import_frame(
    frame: pl.DataFrame,
    *,
    allow_incomplete: bool = False,
) -> None:
    _validate_exact_schema(
        frame,
        reference_review_decision_import_schema(),
        "review decision import",
    )
    versions = set(frame["import_schema_version"].drop_nulls().to_list())
    if versions - {REFERENCE_REVIEW_DECISION_IMPORT_SCHEMA_VERSION}:
        raise ValueError("review decision import has an unsupported schema version")
    if allow_incomplete:
        return
    for row in frame.iter_rows(named=True):
        _normalize_raw_decision(row)


def _validate_inventory_foreign_keys(
    *,
    object_by_id: Mapping[str, Mapping[str, object]],
    candidate_by_id: Mapping[str, Mapping[str, object]],
    observation_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    for media_id, obj in object_by_id.items():
        candidate = candidate_by_id.get(media_id)
        if candidate is None:
            raise ValueError(f"media object {media_id} has no source candidate")
        observation_id = str(candidate["reference_observation_id"])
        observation = observation_by_id.get(observation_id)
        if observation is None:
            raise ValueError(
                f"media candidate {media_id} has no source observation {observation_id}"
            )
        _validate_media_provenance(candidate, observation, obj)
    for media_id, candidate in candidate_by_id.items():
        observation_id = str(candidate["reference_observation_id"])
        observation = observation_by_id.get(observation_id)
        if observation is None:
            raise ValueError(
                f"media candidate {media_id} has no source observation {observation_id}"
            )
        if candidate["source"] != observation["source"]:
            raise ValueError("media candidate and observation sources do not match")


def _validate_exact_schema(
    frame: pl.DataFrame,
    schema: Mapping[str, pl.DataType],
    artifact: str,
) -> None:
    if frame.columns != list(schema):
        missing = sorted(set(schema) - set(frame.columns))
        unknown = sorted(set(frame.columns) - set(schema))
        raise ValueError(
            f"{artifact} columns do not match the schema; "
            f"missing={missing}, unknown={unknown}"
        )
    actual = frame.schema
    mismatches = {
        field: (actual[field], dtype)
        for field, dtype in schema.items()
        if actual[field] != dtype
    }
    if mismatches:
        raise ValueError(f"{artifact} physical schema mismatch: {mismatches}")


def _validate_selection_provenance(
    selection: Mapping[str, object],
    candidate: Mapping[str, object],
    observation: Mapping[str, object],
    obj: Mapping[str, object],
) -> None:
    media_id = str(selection["reference_media_id"])
    if (
        candidate["reference_media_id"] != media_id
        or obj["reference_media_id"] != media_id
    ):
        raise ValueError("selected media provenance is inconsistent")
    observation_id = selection["reference_observation_id"]
    if (
        candidate["reference_observation_id"] != observation_id
        or observation["reference_observation_id"] != observation_id
    ):
        raise ValueError("selected observation provenance is inconsistent")
    if not (selection["source"] == candidate["source"] == observation["source"]):
        raise ValueError("selected source provenance is inconsistent")
    if selection["candidate_accepted_taxon_key"] != observation["accepted_taxon_key"]:
        raise ValueError("selected taxon key does not match its source observation")
    if selection["scientific_name"] != observation["reconciled_scientific_name"]:
        raise ValueError(
            "selected scientific name does not match its source observation"
        )
    for field in ("source_snapshot_version", "licence"):
        if selection[field] != candidate[field]:
            raise ValueError(f"selected media has stale {field}")
    if candidate["source_snapshot_version"] != observation["source_snapshot_version"]:
        raise ValueError(
            "media candidate snapshot does not match its source observation"
        )
    _validate_media_provenance(candidate, observation, obj)


def _validate_media_provenance(
    candidate: Mapping[str, object],
    observation: Mapping[str, object],
    obj: Mapping[str, object],
) -> None:
    if candidate["reference_media_id"] != obj["reference_media_id"]:
        raise ValueError("media candidate and object IDs do not match")
    if candidate["reference_observation_id"] != observation["reference_observation_id"]:
        raise ValueError("media candidate and observation IDs do not match")
    if candidate["source"] != observation["source"]:
        raise ValueError("media candidate and observation sources do not match")
    if candidate["source_snapshot_version"] != observation["source_snapshot_version"]:
        raise ValueError(
            "media candidate snapshot does not match its source observation"
        )


def _validate_duplicate_relationship_foreign_keys(
    relationship: Mapping[str, object],
    *,
    object_by_id: Mapping[str, Mapping[str, object]],
    candidate_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    for side in ("left", "right"):
        media_id = str(relationship[f"{side}_reference_media_id"])
        obj = object_by_id.get(media_id)
        candidate = candidate_by_id.get(media_id)
        if obj is None or candidate is None:
            raise ValueError(
                "duplicate relationship references unknown media provenance"
            )
        if obj["duplicate_group_id"] != relationship["duplicate_group_id"]:
            raise ValueError(
                "duplicate relationship group does not match its media object"
            )
        if (
            obj["canonical_reference_media_id"]
            != relationship["canonical_reference_media_id"]
        ):
            raise ValueError("duplicate relationship canonical media is inconsistent")
        if (
            candidate["reference_observation_id"]
            != relationship[f"{side}_reference_observation_id"]
            or candidate["source"] != relationship[f"{side}_source"]
            or candidate["provider_media_id"]
            != relationship[f"{side}_provider_media_id"]
        ):
            raise ValueError(
                "duplicate relationship endpoint provenance is inconsistent"
            )


def _validate_duplicate_relationship_completeness(
    media_objects: pl.DataFrame,
    relationships_by_group: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    valid = media_objects.filter(pl.col("decode_status") == "valid")
    for key, group in valid.partition_by("duplicate_group_id", as_dict=True).items():
        group_id = str(key[0] if isinstance(key, tuple) else key)
        media_ids = set(group["reference_media_id"])
        expected_group_id = _duplicate_group_id(media_ids)
        if group_id != expected_group_id:
            raise ValueError(
                f"duplicate group ID does not match supplied membership: {group_id}"
            )
        if len(media_ids) == 1:
            if relationships_by_group.get(group_id):
                raise ValueError(
                    "a singleton duplicate group cannot have relationships"
                )
            continue
        relationships = relationships_by_group.get(group_id, ())
        if not relationships:
            raise ValueError(
                f"multi-object duplicate group {group_id} has no relationship ledger"
            )
        adjacency: dict[str, set[str]] = {media_id: set() for media_id in media_ids}
        for relationship in relationships:
            left = str(relationship["left_reference_media_id"])
            right = str(relationship["right_reference_media_id"])
            adjacency[left].add(right)
            adjacency[right].add(left)
        visited: set[str] = set()
        pending = [min(media_ids)]
        while pending:
            media_id = pending.pop()
            if media_id in visited:
                continue
            visited.add(media_id)
            pending.extend(sorted(adjacency[media_id] - visited))
        if visited != media_ids:
            raise ValueError(
                f"duplicate relationship ledger is disconnected for group {group_id}"
            )


def _validate_duplicate_group_taxonomy(
    media_ids: Sequence[str],
    *,
    candidate_by_id: Mapping[str, Mapping[str, object]],
    observation_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    identities: set[tuple[object, object]] = set()
    for media_id in media_ids:
        candidate = candidate_by_id[media_id]
        observation = observation_by_id[str(candidate["reference_observation_id"])]
        identities.add(
            (
                observation["accepted_taxon_key"],
                observation["reconciled_scientific_name"],
            )
        )
    if len(identities) != 1:
        raise ValueError("duplicate group has conflicting taxon provenance")


def _normalize_frame(
    frame: pl.DataFrame,
    *,
    schema: Mapping[str, pl.DataType],
    constructor: Callable[[Sequence[Mapping[str, object]]], pl.DataFrame],
    artifact: str,
) -> pl.DataFrame:
    _validate_exact_schema(frame, schema, artifact)
    return constructor(frame.to_dicts())


def _queue_exclusion_reason(
    candidate: Mapping[str, object],
    obj: Mapping[str, object],
    *,
    include_research_only: bool,
) -> str | None:
    if obj["decode_status"] != "valid":
        return "invalid_or_unavailable_object"
    if candidate["download_status"] not in {"pending", "complete"}:
        return "source_candidate_not_download_eligible"
    if candidate["verification_status"] == "rejected":
        return "provider_rejected"
    if candidate["exclusion_reason"] is not None:
        return "provider_excluded"
    candidate_policy_status = str(candidate["licence_policy_status"])
    object_policy_status = str(obj["licence_policy_status"])
    evaluated_policy_status = _LICENCE_POLICY.evaluate(
        media_licence=candidate["licence"],
        licence_uri=candidate["licence_uri"],
        attribution=candidate["attribution"],
    ).status
    if candidate_policy_status in {"denied", "quarantined"}:
        return "licence_provenance_mismatch"
    if evaluated_policy_status != object_policy_status:
        return "licence_provenance_mismatch"
    if (
        candidate_policy_status in {"allowed", "research_only"}
        and candidate_policy_status != object_policy_status
    ):
        return "licence_provenance_mismatch"
    allowed_licences = {"allowed"}
    if include_research_only:
        allowed_licences.add("research_only")
    if object_policy_status not in allowed_licences:
        return "licence_not_queueable"
    return None


def _unique_rows(
    frame: pl.DataFrame,
    key: str,
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for row in frame.iter_rows(named=True):
        value = str(row[key])
        if value in rows:
            raise ValueError(f"duplicate {key}: {value}")
        rows[value] = dict(row)
    return rows


def _single_proposal(
    values: Sequence[object] | Any,
    *,
    choices: frozenset[str],
) -> str | None:
    normalized = {
        str(value) for value in values if value is not None and str(value) in choices
    }
    return next(iter(normalized)) if len(normalized) == 1 else None


def _strict_frame(
    rows: Sequence[Mapping[str, object]],
    *,
    schema: Mapping[str, pl.DataType],
    sort_by: Sequence[str],
) -> pl.DataFrame:
    materialized = list(rows)
    if not materialized:
        return pl.DataFrame(schema=schema)
    fields = set(schema)
    unknown = sorted(set().union(*(set(row) for row in materialized)) - fields)
    missing = sorted(fields - set.intersection(*(set(row) for row in materialized)))
    if unknown or missing:
        raise ValueError(
            f"review rows do not match schema; missing={missing}, unknown={unknown}"
        )
    return pl.DataFrame(materialized, schema=schema, strict=True).sort(sort_by)


def _resolution_timestamp(
    queue: pl.DataFrame,
    decisions: pl.DataFrame,
    supplied: datetime | None,
) -> datetime:
    if supplied is not None:
        return _utc_datetime(supplied, field="resolved_at")
    values: list[datetime] = []
    values.extend(value for value in queue["created_at"].to_list() if value is not None)
    values.extend(
        value for value in decisions["reviewed_at"].to_list() if value is not None
    )
    return max(values, default=datetime(1970, 1, 1, tzinfo=UTC))


def _default_queue_created_at(
    selections: pl.DataFrame,
    media_objects: pl.DataFrame,
) -> datetime:
    values: list[datetime] = []
    if "selected_at" in selections.columns:
        values.extend(
            value for value in selections["selected_at"].to_list() if value is not None
        )
    if (
        "downloaded_at" in media_objects.columns
        and "reference_media_id" in selections.columns
    ):
        selected_media_ids = set(selections["reference_media_id"].to_list())
        selected_group_ids = {
            row["duplicate_group_id"]
            for row in media_objects.select(
                "reference_media_id", "duplicate_group_id"
            ).iter_rows(named=True)
            if row["reference_media_id"] in selected_media_ids
            and row["duplicate_group_id"] is not None
        }
        values.extend(
            row["downloaded_at"]
            for row in media_objects.select(
                "reference_media_id", "duplicate_group_id", "downloaded_at"
            ).iter_rows(named=True)
            if (
                row["reference_media_id"] in selected_media_ids
                or row["duplicate_group_id"] in selected_group_ids
            )
            and row["downloaded_at"] is not None
        )
    if not values:
        return datetime(1970, 1, 1, tzinfo=UTC)
    return max(_utc_datetime(value, field="input timestamp") for value in values)


def _duplicate_group_id(media_ids: Sequence[str] | set[str]) -> str:
    digest = _payload_hash(
        {"reference_media_ids": sorted(str(value) for value in media_ids)}
    ).removeprefix("sha256:")
    return f"reference-duplicate-group:{digest[:32]}"


def _conflict_group_id(
    request_id: str,
    decision_ids: Sequence[str],
    fields: Sequence[str],
) -> str:
    return "reference-review-conflict:" + _payload_hash(
        {
            "review_request_id": request_id,
            "effective_decision_ids": sorted(decision_ids),
            "conflicting_fields": sorted(fields),
        }
    ).removeprefix("sha256:")


def _frame_fingerprint(frame: pl.DataFrame) -> str:
    return _payload_hash(
        {
            "schema": [(name, str(dtype)) for name, dtype in frame.schema.items()],
            "rows": [_jsonable(row) for row in frame.iter_rows(named=True)],
        }
    )


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, datetime):
        return _datetime_text(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
    return value


def _json_scalar(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _utc_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _datetime_text(value: datetime) -> str:
    return _utc_datetime(value, field="datetime").isoformat(timespec="microseconds")


def _required_canonical_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be nonblank text")
    if value != value.strip():
        raise ValueError(f"{field} must not have surrounding whitespace")
    return value


def _optional_canonical_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_canonical_text(value, field=field)


def _canonical_reviewer_id(value: object) -> str:
    actor = _required_canonical_text(value, field="verified_by")
    if _REVIEWER_ID_PATTERN.fullmatch(actor) is None:
        raise ValueError(
            "verified_by must be a canonical lowercase ASCII reviewer identifier"
        )
    return actor


def _choice(
    value: object,
    *,
    field: str,
    choices: frozenset[str],
) -> str:
    normalized = _required_canonical_text(value, field=field)
    if normalized not in choices:
        raise ValueError(f"unsupported {field}: {normalized}")
    return normalized


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _local_artifact_record(
    path: Path,
    *,
    final_directory: Path,
) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "uri": str((final_directory / path.name).resolve()),
        "committed": True,
        "byte_count": len(content),
        "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
    }


def _review_markdown(report: Mapping[str, object], *, title: str) -> str:
    counts = report.get("counts")
    status_counts = report.get("review_status_counts")
    artifacts = report.get("artifacts")
    lines = [
        f"# {title}",
        "",
        f"- Command: `{report.get('command', 'not_instrumented')}`",
        f"- Run ID: `{report.get('run_id', 'not_instrumented')}`",
        f"- PID: `{report.get('pid', 'not_instrumented')}`",
        f"- Git SHA: `{report.get('git_sha') or 'not_instrumented'}`",
        f"- Status: `{report.get('status', 'not_instrumented')}`",
        f"- Started: `{report.get('started_at', 'not_instrumented')}`",
        f"- Ended: `{report.get('ended_at', 'not_instrumented')}`",
        f"- Elapsed seconds: `{report.get('elapsed_seconds', 'not_instrumented')}`",
        "",
        "## Counts",
    ]
    if isinstance(counts, Mapping):
        lines.extend(f"- `{key}`: `{value}`" for key, value in sorted(counts.items()))
    else:
        lines.append("- `not_instrumented`")
    if isinstance(status_counts, Mapping):
        lines.extend(["", "## Review statuses"])
        lines.extend(
            f"- `{key}`: `{value}`" for key, value in sorted(status_counts.items())
        )
    lines.extend(["", "## Artifacts"])
    if isinstance(artifacts, Mapping) and artifacts:
        for key, record in sorted(artifacts.items()):
            uri = record.get("uri") if isinstance(record, Mapping) else None
            lines.append(f"- `{key}`: `{uri or 'not_instrumented'}`")
    else:
        lines.append("- `not_instrumented`")
    lines.append("")
    return "\n".join(lines)


def _log_event(event: str, **fields: object) -> None:
    _LOGGER.info(
        json.dumps(
            {"event": event, **fields},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    )


__all__ = [
    "REFERENCE_REVIEW_CONFLICTS_FILE",
    "REFERENCE_REVIEW_CONFLICTS_SCHEMA_VERSION",
    "REFERENCE_REVIEW_DECISION_IMPORT_SCHEMA_VERSION",
    "REFERENCE_REVIEW_DECISION_IMPORT_FILE",
    "REFERENCE_REVIEW_DECISION_TEMPLATE_FILE",
    "REFERENCE_REVIEW_EXCLUDED_FILE",
    "REFERENCE_REVIEW_HISTORY_HEAD_SCHEMA_VERSION",
    "REFERENCE_REVIEW_HISTORY_SCHEMA_VERSION",
    "REFERENCE_REVIEW_OUTCOMES_FILE",
    "REFERENCE_REVIEW_OUTCOMES_SCHEMA_VERSION",
    "REFERENCE_REVIEW_QUEUE_PROVENANCE_FILE",
    "REFERENCE_REVIEW_QUEUE_PROVENANCE_SCHEMA_VERSION",
    "REFERENCE_REVIEW_RESOLVED_MEDIA_SCHEMA_VERSION",
    "REFERENCE_REVIEW_VERIFIED_FILE",
    "ReferenceReviewQueueResult",
    "ReferenceReviewWorkflowResult",
    "advance_reference_review_history_head",
    "build_reference_review_queue",
    "import_reference_review_decisions",
    "initialize_reference_review_history_head",
    "reference_review_conflict_schema",
    "reference_review_decision_import_schema",
    "reference_review_decision_template",
    "reference_review_outcome_schema",
    "reference_review_queue_provenance_schema",
    "reference_review_resolved_media_schema",
    "resolve_reference_review_statuses",
    "select_verified_reference_media",
    "validate_reference_review_history_head",
    "validate_reference_review_history_head_destination",
    "validate_reference_review_packet_artifact",
    "validate_reference_review_queue_source_bindings",
    "write_reference_review_export",
    "write_reference_review_import",
]
