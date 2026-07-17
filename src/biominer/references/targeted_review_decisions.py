"""Append strict human decisions for statistically targeted references."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.review import (
    REFERENCE_REVIEW_DECISION_IMPORT_SCHEMA_VERSION,
    ReferenceReviewWorkflowResult,
    append_reference_review_decisions,
    project_reference_review_queue_provenance,
    reference_review_decision_import_schema,
)
from biominer.references.schemas import (
    REFERENCE_LIFE_STAGES,
    REFERENCE_REVIEW_CONFIDENCE_VALUES,
    REFERENCE_VIEWS,
    REFERENCE_VISUAL_DOMAINS,
    reference_review_decisions_frame,
    reference_review_queue_schema,
    validate_reference_review_decisions,
    validate_reference_review_queue,
)
from biominer.references.targeted_review import (
    validate_targeted_reference_review_queue,
)
from biominer.storage.parquet import write_parquet


TARGETED_REFERENCE_REVIEW_DECISIONS_FILE = (
    "targeted_reference_review_decisions.parquet"
)
TARGETED_REFERENCE_REVIEW_DECISION_BINDINGS_FILE = (
    "targeted_reference_review_decision_bindings.parquet"
)
TARGETED_REFERENCE_REVIEW_DECISION_SCHEMA_VERSION = (
    "targeted-reference-review-decisions-v1.0.0"
)
TARGETED_REFERENCE_REVIEW_DECISION_BINDING_SCHEMA_VERSION = (
    "targeted-reference-review-decision-bindings-v1.0.0"
)
TARGETED_REVIEW_ACTIONS = frozenset(
    {"verify", "exclude", "uncertain", "request_second_review"}
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


def targeted_reference_review_decision_schema() -> dict[str, pl.DataType]:
    return {
        "targeted_decision_schema_version": pl.String,
        "review_request_id": pl.String,
        "reference_media_id": pl.String,
        "targeting_fingerprint": pl.String,
        "review_action": pl.String,
        "review_round": pl.UInt16,
        "verified_by": pl.String,
        "reviewed_at": pl.Datetime("us", "UTC"),
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "view": pl.String,
        "review_confidence": pl.String,
        "review_notes": pl.String,
        "exclusion_reason": pl.String,
        "alternative_species": pl.String,
        "conflicts_with_decision_id": pl.String,
        "targeted_decision_fingerprint": pl.String,
    }


def targeted_reference_review_decision_binding_schema() -> (
    dict[str, pl.DataType]
):
    return {
        "schema_version": pl.String,
        "targeted_decision_fingerprint": pl.String,
        "review_decision_id": pl.String,
        "review_request_id": pl.String,
        "reference_media_id": pl.String,
        "targeting_fingerprint": pl.String,
        "review_action": pl.String,
        "alternative_species": pl.String,
        "binding_fingerprint": pl.String,
    }


@dataclass(frozen=True, slots=True)
class TargetedReferenceReviewResult:
    targeted_queue: pl.DataFrame
    targeted_decisions: pl.DataFrame
    decision_bindings: pl.DataFrame
    workflow: ReferenceReviewWorkflowResult
    imported_decision_count: int
    idempotent_replay_count: int


def targeted_reference_review_decisions_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for source in rows or []:
        row = dict(source)
        row.setdefault(
            "targeted_decision_schema_version",
            TARGETED_REFERENCE_REVIEW_DECISION_SCHEMA_VERSION,
        )
        row["targeted_decision_fingerprint"] = ""
        payload = dict(row)
        payload.pop("targeted_decision_fingerprint")
        row["targeted_decision_fingerprint"] = canonical_semantic_fingerprint(
            payload
        )
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=targeted_reference_review_decision_schema(),
        orient="row",
        strict=True,
    ).sort(
        "reference_media_id",
        "review_round",
        "verified_by",
        "reviewed_at",
        "targeted_decision_fingerprint",
    )
    validate_targeted_reference_review_decisions(frame)
    return frame


def validate_targeted_reference_review_decisions(frame: pl.DataFrame) -> None:
    if frame.schema != targeted_reference_review_decision_schema():
        raise ValueError("targeted reference decision schema mismatch")
    if frame["targeted_decision_fingerprint"].n_unique() != frame.height:
        raise ValueError("targeted reference decisions repeat a source decision")
    expected = frame.sort(
        "reference_media_id",
        "review_round",
        "verified_by",
        "reviewed_at",
        "targeted_decision_fingerprint",
    )
    if not frame.equals(expected):
        raise ValueError("targeted reference decisions are not deterministically sorted")
    for row in frame.iter_rows(named=True):
        if (
            row["targeted_decision_schema_version"]
            != TARGETED_REFERENCE_REVIEW_DECISION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported targeted reference decision version")
        action = _choice(
            row["review_action"],
            field="review_action",
            choices=TARGETED_REVIEW_ACTIONS,
        )
        _required_text(row["review_request_id"], field="review_request_id")
        _required_text(row["reference_media_id"], field="reference_media_id")
        _full_sha256(row["targeting_fingerprint"], field="targeting_fingerprint")
        _positive_int(row["review_round"], field="review_round")
        _required_text(row["verified_by"], field="verified_by")
        _choice(row["life_stage"], field="life_stage", choices=REFERENCE_LIFE_STAGES)
        _choice(
            row["visual_domain"],
            field="visual_domain",
            choices=REFERENCE_VISUAL_DOMAINS,
        )
        _choice(row["view"], field="view", choices=REFERENCE_VIEWS)
        _choice(
            row["review_confidence"],
            field="review_confidence",
            choices=REFERENCE_REVIEW_CONFIDENCE_VALUES,
        )
        notes = _optional_text(row["review_notes"], field="review_notes")
        exclusion = _optional_text(
            row["exclusion_reason"],
            field="exclusion_reason",
        )
        alternative = _optional_text(
            row["alternative_species"],
            field="alternative_species",
        )
        _optional_text(
            row["conflicts_with_decision_id"],
            field="conflicts_with_decision_id",
        )
        if action == "verify" and (exclusion is not None or alternative is not None):
            raise ValueError(
                "verified target decisions cannot exclude or name an alternative"
            )
        if action == "exclude" and exclusion is None:
            raise ValueError("excluded target decisions require an exclusion reason")
        if action == "uncertain" and notes is None:
            raise ValueError("uncertain target decisions require review notes")
        if action in {"uncertain", "request_second_review"} and exclusion is not None:
            raise ValueError("non-decisive target decisions cannot exclude a reference")
        payload = dict(row)
        fingerprint = payload.pop("targeted_decision_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("targeted reference decision fingerprint mismatch")


def review_statistically_flagged_support(
    targeted_queue: pl.DataFrame,
    queue_provenance: pl.DataFrame,
    targeted_decisions: pl.DataFrame,
    *,
    existing_decisions: pl.DataFrame | None = None,
    resolved_at: datetime | None = None,
) -> TargetedReferenceReviewResult:
    """Bind targeted decisions to the append-only reference review ledger."""

    validate_targeted_reference_review_queue(targeted_queue)
    validate_targeted_reference_review_decisions(targeted_decisions)
    queue = targeted_queue.select(list(reference_review_queue_schema())).sort(
        "review_priority",
        "reference_media_id",
        "review_request_id",
    )
    validate_reference_review_queue(queue)
    projected_provenance = project_reference_review_queue_provenance(
        queue_provenance,
        queue,
    )
    targeted_by_request = {
        str(row["review_request_id"]): row
        for row in targeted_queue.iter_rows(named=True)
    }
    _validate_targeted_decision_bindings(targeted_decisions, targeted_by_request)
    raw_import = _base_decision_import(targeted_decisions)
    existing = (
        existing_decisions
        if existing_decisions is not None
        else reference_review_decisions_frame([])
    )
    validate_reference_review_decisions(existing)
    appended = append_reference_review_decisions(
        raw_import,
        queue=queue,
        queue_provenance=projected_provenance,
        existing_decisions=existing,
        resolved_at=resolved_at,
    )
    bindings = _decision_bindings(
        targeted_decisions,
        appended.workflow.decisions,
    )
    result = TargetedReferenceReviewResult(
        targeted_queue=targeted_queue,
        targeted_decisions=targeted_decisions,
        decision_bindings=bindings,
        workflow=appended.workflow,
        imported_decision_count=appended.imported_decision_count,
        idempotent_replay_count=appended.idempotent_replay_count,
    )
    _validate_targeted_result(result, existing_decisions=existing)
    return result


def write_targeted_reference_review_result(
    result: TargetedReferenceReviewResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    _validate_targeted_result(
        result,
        existing_decisions=reference_review_decisions_frame([]),
        require_existing_subset=False,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "targeted_decisions": write_parquet(
            result.targeted_decisions,
            root / TARGETED_REFERENCE_REVIEW_DECISIONS_FILE,
        ),
        "decision_bindings": write_parquet(
            result.decision_bindings,
            root / TARGETED_REFERENCE_REVIEW_DECISION_BINDINGS_FILE,
        ),
        "decision_ledger": write_parquet(
            result.workflow.decisions,
            root / "reference_review_decisions.parquet",
        ),
        "outcomes": write_parquet(
            result.workflow.outcomes,
            root / "reference_review_outcomes.parquet",
        ),
        "verified": write_parquet(
            result.workflow.verified,
            root / "verified_reference_media.parquet",
        ),
        "excluded": write_parquet(
            result.workflow.excluded,
            root / "excluded_reference_media.parquet",
        ),
    }
    return artifacts


def _base_decision_import(targeted: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for row in targeted.iter_rows(named=True):
        action = str(row["review_action"])
        notes = _base_review_notes(row)
        rows.append(
            {
                "import_schema_version": (
                    REFERENCE_REVIEW_DECISION_IMPORT_SCHEMA_VERSION
                ),
                "review_request_id": row["review_request_id"],
                "reference_media_id": row["reference_media_id"],
                "review_round": row["review_round"],
                "verified_by": row["verified_by"],
                "reviewed_at": row["reviewed_at"],
                "target_identity_verified": (
                    True
                    if action == "verify"
                    else False
                    if action == "exclude"
                    else None
                ),
                "verification_status": (
                    "verified"
                    if action == "verify"
                    else "excluded"
                    if action == "exclude"
                    else "uncertain"
                ),
                "life_stage": row["life_stage"],
                "visual_domain": row["visual_domain"],
                "view": row["view"],
                "review_confidence": row["review_confidence"],
                "review_notes": notes,
                "exclusion_reason": row["exclusion_reason"],
                "conflicts_with_decision_id": row[
                    "conflicts_with_decision_id"
                ],
            }
        )
    return pl.DataFrame(
        rows,
        schema=reference_review_decision_import_schema(),
        orient="row",
        strict=True,
    ).sort(
        "reference_media_id",
        "review_round",
        "verified_by",
        "reviewed_at",
        "review_request_id",
    )


def _base_review_notes(row: Mapping[str, object]) -> str | None:
    notes = [str(row["review_notes"])] if row["review_notes"] is not None else []
    if row["review_action"] == "request_second_review":
        notes.append("Second review requested.")
    if row["alternative_species"] is not None:
        notes.append(f"Alternative species noted: {row['alternative_species']}.")
    return " ".join(notes) or None


def _validate_targeted_decision_bindings(
    decisions: pl.DataFrame,
    queue_by_request: Mapping[str, Mapping[str, object]],
) -> None:
    for row in decisions.iter_rows(named=True):
        request_id = str(row["review_request_id"])
        queue = queue_by_request.get(request_id)
        if queue is None:
            raise ValueError(
                f"targeted decision references unknown review request {request_id}"
            )
        if row["reference_media_id"] != queue["reference_media_id"]:
            raise ValueError("targeted decision media does not match its request")
        if row["targeting_fingerprint"] != queue["targeting_fingerprint"]:
            raise ValueError("targeted decision has a stale targeting fingerprint")
        alternative = row["alternative_species"]
        if alternative is not None and str(alternative).casefold() == str(
            queue["scientific_name"]
        ).casefold():
            raise ValueError("alternative species must differ from the target species")


def _decision_bindings(
    targeted: pl.DataFrame,
    ledger: pl.DataFrame,
) -> pl.DataFrame:
    ledger_by_vote = {
        (
            str(row["review_request_id"]),
            str(row["reference_media_id"]),
            int(row["review_round"]),
            str(row["verified_by"]),
            row["reviewed_at"],
        ): row
        for row in ledger.iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    for source in targeted.iter_rows(named=True):
        key = (
            str(source["review_request_id"]),
            str(source["reference_media_id"]),
            int(source["review_round"]),
            str(source["verified_by"]),
            source["reviewed_at"],
        )
        decision = ledger_by_vote.get(key)
        if decision is None:
            raise ValueError("targeted decision is missing from the append-only ledger")
        row = {
            "schema_version": (
                TARGETED_REFERENCE_REVIEW_DECISION_BINDING_SCHEMA_VERSION
            ),
            "targeted_decision_fingerprint": source[
                "targeted_decision_fingerprint"
            ],
            "review_decision_id": decision["review_decision_id"],
            "review_request_id": source["review_request_id"],
            "reference_media_id": source["reference_media_id"],
            "targeting_fingerprint": source["targeting_fingerprint"],
            "review_action": source["review_action"],
            "alternative_species": source["alternative_species"],
            "binding_fingerprint": "",
        }
        payload = dict(row)
        payload.pop("binding_fingerprint")
        row["binding_fingerprint"] = canonical_semantic_fingerprint(payload)
        rows.append(row)
    return pl.DataFrame(
        rows,
        schema=targeted_reference_review_decision_binding_schema(),
        orient="row",
        strict=True,
    ).sort("reference_media_id", "targeted_decision_fingerprint")


def _validate_targeted_result(
    result: TargetedReferenceReviewResult,
    *,
    existing_decisions: pl.DataFrame,
    require_existing_subset: bool = True,
) -> None:
    validate_targeted_reference_review_queue(result.targeted_queue)
    validate_targeted_reference_review_decisions(result.targeted_decisions)
    if (
        result.decision_bindings.schema
        != targeted_reference_review_decision_binding_schema()
        or result.decision_bindings.height != result.targeted_decisions.height
    ):
        raise ValueError("targeted decision bindings are incomplete")
    ledger_by_id = {
        str(row["review_decision_id"]): row
        for row in result.workflow.decisions.iter_rows(named=True)
    }
    target_fingerprints = set(
        result.targeted_decisions["targeted_decision_fingerprint"]
    )
    for row in result.decision_bindings.iter_rows(named=True):
        if (
            row["schema_version"]
            != TARGETED_REFERENCE_REVIEW_DECISION_BINDING_SCHEMA_VERSION
            or row["targeted_decision_fingerprint"] not in target_fingerprints
            or str(row["review_decision_id"]) not in ledger_by_id
        ):
            raise ValueError("targeted decision binding is inconsistent")
        payload = dict(row)
        fingerprint = payload.pop("binding_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("targeted decision binding fingerprint mismatch")
    if require_existing_subset:
        current = {
            str(row["review_decision_id"]): row
            for row in result.workflow.decisions.iter_rows(named=True)
        }
        for row in existing_decisions.iter_rows(named=True):
            if current.get(str(row["review_decision_id"])) != row:
                raise ValueError("append-only review changed an existing decision")


def _choice(value: object, *, field: str, choices: frozenset[str]) -> str:
    text = _required_text(value, field=field)
    if text not in choices:
        raise ValueError(f"unsupported {field}: {text}")
    return text


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text or text != value:
        raise ValueError(f"{field} must be canonical nonblank text")
    return text


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _full_sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full sha256 fingerprint")
    return text


__all__ = [
    "TARGETED_REFERENCE_REVIEW_DECISIONS_FILE",
    "TARGETED_REFERENCE_REVIEW_DECISION_BINDINGS_FILE",
    "TARGETED_REFERENCE_REVIEW_DECISION_BINDING_SCHEMA_VERSION",
    "TARGETED_REFERENCE_REVIEW_DECISION_SCHEMA_VERSION",
    "TARGETED_REVIEW_ACTIONS",
    "TargetedReferenceReviewResult",
    "review_statistically_flagged_support",
    "targeted_reference_review_decision_binding_schema",
    "targeted_reference_review_decision_schema",
    "targeted_reference_review_decisions_frame",
    "validate_targeted_reference_review_decisions",
    "write_targeted_reference_review_result",
]
