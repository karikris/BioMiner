"""Mutually explicit release, screening and unresolved dynamic-pool outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.flickr_export import validate_verified_flickr_export
from biominer.evaluation.flickr_release import (
    FlickrReleaseDecision,
    FlickrReleaseState,
)


DYNAMIC_POOL_OUTCOME_EVIDENCE_VERSION = "dynamic-pool-outcome-evidence-v1.0.0"
HUMAN_REVIEWED_RELEASE_SCHEMA_VERSION = "human-reviewed-release-lane-v1.0.0"
HUMAN_REVIEWED_RELEASE_LABEL = "human_reviewed_release_candidate"
SCREENING_THRESHOLD_STATUSES = frozenset({"not_evaluated", "selected", "infeasible"})
CONFLICT_STATUSES = frozenset({"not_required", "resolved", "pending", "unresolved"})
HUMAN_REVIEW_DECISIONS = frozenset({"include", "exclude", "uncertain"})

HUMAN_REVIEWED_RELEASE_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "lane_fingerprint": pl.String,
    "lane_row_fingerprint": pl.String,
    "source_evidence_fingerprint": pl.String,
    "outcome_lane": pl.String,
    "outcome_label": pl.String,
    "item_id": pl.String,
    "source_record_id": pl.String,
    "candidate_species_key": pl.String,
    "route": pl.String,
    "source_image_sha256": pl.String,
    "human_review_decision": pl.String,
    "review_decision_fingerprint": pl.String,
    "review_source_image_sha256": pl.String,
    "conflict_status": pl.String,
    "occurrence_claim_supported": pl.Boolean,
    "eligible_for_final_occurrence_dataset": pl.Boolean,
    "release_state": pl.String,
    "release_reasons": pl.List(pl.String),
    "release_authorized": pl.Boolean,
    "human_reviewed": pl.Boolean,
    "model_evidence_authorizes_release": pl.Boolean,
    "calibrated_supported_probability": pl.Float64,
    "evidence_model_fingerprint": pl.String,
    "calibrator_fingerprint": pl.String,
    "split_fingerprint": pl.String,
    "screening_threshold_selection_fingerprint": pl.String,
}


@dataclass(frozen=True, slots=True)
class DynamicPoolOutcomeEvidence:
    """One candidate's review, release, calibration and triage evidence."""

    item_id: str
    source_record_id: str
    source_image_sha256: str
    candidate_species_key: str
    route: str
    evidence_model_fingerprint: str
    calibrator_fingerprint: str
    split_fingerprint: str
    release_decision: FlickrReleaseDecision
    conflict_status: str
    occurrence_claim_supported: bool
    screening_threshold_status: str
    route_compatible: bool
    reference_coverage_sufficient: bool
    geographic_evidence_sufficient: bool
    visual_detail_sufficient: bool
    domain_negative_absent: bool
    out_of_distribution_absent: bool
    review_priority: float
    human_review_decision: str | None = None
    review_decision_fingerprint: str | None = None
    review_source_image_sha256: str | None = None
    calibrated_supported_probability: float | None = None
    screening_threshold_selection_fingerprint: str | None = None
    screening_threshold: float | None = None
    triage_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("item_id", "source_record_id", "candidate_species_key", "route"):
            object.__setattr__(
                self, field, _required_text(getattr(self, field), field=field)
            )
        for field in (
            "source_image_sha256",
            "evidence_model_fingerprint",
            "calibrator_fingerprint",
            "split_fingerprint",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field=field))
        if not isinstance(self.release_decision, FlickrReleaseDecision):
            raise TypeError("release_decision must be a FlickrReleaseDecision")
        if self.release_decision.source_record_id != self.source_record_id:
            raise ValueError("release decision references another source record")
        if self.release_decision.source_image_sha256 != self.source_image_sha256:
            raise ValueError("release decision references another source image")
        decision = _optional_choice(
            self.human_review_decision,
            field="human_review_decision",
            allowed=HUMAN_REVIEW_DECISIONS,
        )
        decision_fingerprint = _optional_sha256(
            self.review_decision_fingerprint,
            field="review_decision_fingerprint",
        )
        review_hash = _optional_sha256(
            self.review_source_image_sha256,
            field="review_source_image_sha256",
        )
        if decision is None:
            if decision_fingerprint is not None or review_hash is not None:
                raise ValueError("unreviewed evidence cannot claim review provenance")
        elif decision_fingerprint is None or review_hash is None:
            raise ValueError(
                "reviewed evidence requires decision and source fingerprints"
            )
        object.__setattr__(self, "human_review_decision", decision)
        object.__setattr__(self, "review_decision_fingerprint", decision_fingerprint)
        object.__setattr__(self, "review_source_image_sha256", review_hash)
        conflict = _choice(
            self.conflict_status,
            field="conflict_status",
            allowed=CONFLICT_STATUSES,
        )
        threshold_status = _choice(
            self.screening_threshold_status,
            field="screening_threshold_status",
            allowed=SCREENING_THRESHOLD_STATUSES,
        )
        probability = _optional_probability(
            self.calibrated_supported_probability,
            field="calibrated_supported_probability",
        )
        threshold = _optional_probability(
            self.screening_threshold,
            field="screening_threshold",
        )
        threshold_fingerprint = _optional_sha256(
            self.screening_threshold_selection_fingerprint,
            field="screening_threshold_selection_fingerprint",
        )
        if threshold_status == "selected":
            if threshold is None or threshold_fingerprint is None:
                raise ValueError("selected screening threshold requires provenance")
        elif threshold is not None or threshold_fingerprint is not None:
            raise ValueError("unselected screening threshold cannot claim parameters")
        object.__setattr__(self, "conflict_status", conflict)
        object.__setattr__(self, "screening_threshold_status", threshold_status)
        object.__setattr__(self, "calibrated_supported_probability", probability)
        object.__setattr__(self, "screening_threshold", threshold)
        object.__setattr__(
            self,
            "screening_threshold_selection_fingerprint",
            threshold_fingerprint,
        )
        for field in (
            "occurrence_claim_supported",
            "route_compatible",
            "reference_coverage_sufficient",
            "geographic_evidence_sufficient",
            "visual_detail_sufficient",
            "domain_negative_absent",
            "out_of_distribution_absent",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be boolean")
        priority = _probability(self.review_priority, field="review_priority")
        object.__setattr__(self, "review_priority", priority)
        reasons = tuple(
            _required_text(value, field="triage_reasons")
            for value in self.triage_reasons
        )
        if len(reasons) != len(set(reasons)):
            raise ValueError("triage_reasons must be unique")
        object.__setattr__(self, "triage_reasons", reasons)
        if self.occurrence_claim_supported and decision != "include":
            raise ValueError("occurrence support requires an include review")
        if self.release_decision.eligible_for_final_occurrence_dataset:
            if (
                decision != "include"
                or review_hash != self.source_image_sha256
                or self.release_decision.state is not FlickrReleaseState.ELIGIBLE
            ):
                raise ValueError("eligible release decision lacks source-bound review")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": DYNAMIC_POOL_OUTCOME_EVIDENCE_VERSION,
                "item_id": self.item_id,
                "source_record_id": self.source_record_id,
                "source_image_sha256": self.source_image_sha256,
                "candidate_species_key": self.candidate_species_key,
                "route": self.route,
                "evidence_model_fingerprint": self.evidence_model_fingerprint,
                "calibrator_fingerprint": self.calibrator_fingerprint,
                "split_fingerprint": self.split_fingerprint,
                "release_decision": {
                    "state": str(self.release_decision.state),
                    "eligible_for_final_occurrence_dataset": (
                        self.release_decision.eligible_for_final_occurrence_dataset
                    ),
                    "reasons": tuple(
                        str(reason) for reason in self.release_decision.reasons
                    ),
                    "source_image_sha256": self.release_decision.source_image_sha256,
                    "review_source_image_sha256": (
                        self.release_decision.review_source_image_sha256
                    ),
                },
                "conflict_status": self.conflict_status,
                "occurrence_claim_supported": self.occurrence_claim_supported,
                "screening_threshold_status": self.screening_threshold_status,
                "screening_threshold_selection_fingerprint": (
                    self.screening_threshold_selection_fingerprint
                ),
                "screening_threshold": self.screening_threshold,
                "calibrated_supported_probability": (
                    self.calibrated_supported_probability
                ),
                "route_compatible": self.route_compatible,
                "reference_coverage_sufficient": (self.reference_coverage_sufficient),
                "geographic_evidence_sufficient": (self.geographic_evidence_sufficient),
                "visual_detail_sufficient": self.visual_detail_sufficient,
                "domain_negative_absent": self.domain_negative_absent,
                "out_of_distribution_absent": self.out_of_distribution_absent,
                "review_priority": self.review_priority,
                "human_review_decision": self.human_review_decision,
                "review_decision_fingerprint": self.review_decision_fingerprint,
                "review_source_image_sha256": self.review_source_image_sha256,
                "triage_reasons": self.triage_reasons,
            }
        )


@dataclass(frozen=True, slots=True)
class DynamicPoolLaneProjection:
    table: pl.DataFrame
    lane_fingerprint: str
    source_item_count: int
    projected_item_count: int


def project_human_reviewed_release_set(
    evidence: Sequence[DynamicPoolOutcomeEvidence],
) -> DynamicPoolLaneProjection:
    """Project only fully reviewed rows already permitted by release policy."""

    items = _normalized_evidence(evidence)
    selected = tuple(item for item in items if _release_lane_eligible(item))
    semantic_rows = [_human_release_row_base(item) for item in selected]
    row_fingerprints = [canonical_semantic_fingerprint(row) for row in semantic_rows]
    lane_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": HUMAN_REVIEWED_RELEASE_SCHEMA_VERSION,
            "source_evidence_fingerprints": [item.fingerprint for item in items],
            "projected_row_fingerprints": row_fingerprints,
            "selection_rule": "all_human_and_release_gates_passed",
            "model_evidence_authorizes_release": False,
        }
    )
    rows = [
        {
            "schema_version": HUMAN_REVIEWED_RELEASE_SCHEMA_VERSION,
            "lane_fingerprint": lane_fingerprint,
            "lane_row_fingerprint": row_fingerprint,
            **base,
        }
        for base, row_fingerprint in zip(semantic_rows, row_fingerprints, strict=True)
    ]
    table = (
        pl.DataFrame(rows, schema=HUMAN_REVIEWED_RELEASE_SCHEMA, strict=True).sort(
            "item_id"
        )
        if rows
        else pl.DataFrame(schema=HUMAN_REVIEWED_RELEASE_SCHEMA)
    )
    validate_human_reviewed_release_set(table)
    return DynamicPoolLaneProjection(
        table=table,
        lane_fingerprint=lane_fingerprint,
        source_item_count=len(items),
        projected_item_count=table.height,
    )


def validate_human_reviewed_release_set(table: pl.DataFrame) -> None:
    """Require every projected row to satisfy the verified Flickr export gate."""

    if not isinstance(table, pl.DataFrame):
        raise TypeError("human-reviewed release set must be a Polars DataFrame")
    if table.schema != HUMAN_REVIEWED_RELEASE_SCHEMA:
        raise ValueError("human-reviewed release schema does not match contract")
    if not table.height:
        return
    if not table.equals(table.sort("item_id")):
        raise ValueError("human-reviewed release set is not sorted")
    if table["item_id"].n_unique() != table.height:
        raise ValueError("human-reviewed release item IDs must be unique")
    if table["source_record_id"].n_unique() != table.height:
        raise ValueError("human-reviewed release source records must be unique")
    if table.filter(
        (pl.col("schema_version") != HUMAN_REVIEWED_RELEASE_SCHEMA_VERSION)
        | (pl.col("outcome_lane") != "human_reviewed_release")
        | (pl.col("outcome_label") != HUMAN_REVIEWED_RELEASE_LABEL)
        | (pl.col("human_review_decision") != "include")
        | ~pl.col("occurrence_claim_supported")
        | ~pl.col("eligible_for_final_occurrence_dataset")
        | (pl.col("release_state") != "eligible")
        | ~pl.col("release_authorized")
        | ~pl.col("human_reviewed")
        | pl.col("model_evidence_authorizes_release")
        | ~pl.col("conflict_status").is_in(["resolved", "not_required"])
        | (pl.col("review_source_image_sha256") != pl.col("source_image_sha256"))
    ).height:
        raise ValueError("human-reviewed release set contains an ineligible row")
    for row in table.iter_rows(named=True):
        base = {
            field: row[field]
            for field in HUMAN_REVIEWED_RELEASE_SCHEMA
            if field
            not in {"schema_version", "lane_fingerprint", "lane_row_fingerprint"}
        }
        if row["lane_row_fingerprint"] != canonical_semantic_fingerprint(base):
            raise ValueError("human-reviewed release row fingerprint mismatch")
    if table["lane_fingerprint"].n_unique() != 1:
        raise ValueError("human-reviewed release set has mixed lane fingerprints")
    validate_verified_flickr_export(table)


def _human_release_row_base(item: DynamicPoolOutcomeEvidence) -> dict[str, object]:
    assert item.human_review_decision == "include"
    assert item.review_decision_fingerprint is not None
    assert item.review_source_image_sha256 is not None
    return {
        "source_evidence_fingerprint": item.fingerprint,
        "outcome_lane": "human_reviewed_release",
        "outcome_label": HUMAN_REVIEWED_RELEASE_LABEL,
        "item_id": item.item_id,
        "source_record_id": item.source_record_id,
        "candidate_species_key": item.candidate_species_key,
        "route": item.route,
        "source_image_sha256": item.source_image_sha256,
        "human_review_decision": item.human_review_decision,
        "review_decision_fingerprint": item.review_decision_fingerprint,
        "review_source_image_sha256": item.review_source_image_sha256,
        "conflict_status": item.conflict_status,
        "occurrence_claim_supported": item.occurrence_claim_supported,
        "eligible_for_final_occurrence_dataset": True,
        "release_state": str(item.release_decision.state),
        "release_reasons": [str(reason) for reason in item.release_decision.reasons],
        "release_authorized": True,
        "human_reviewed": True,
        "model_evidence_authorizes_release": False,
        "calibrated_supported_probability": item.calibrated_supported_probability,
        "evidence_model_fingerprint": item.evidence_model_fingerprint,
        "calibrator_fingerprint": item.calibrator_fingerprint,
        "split_fingerprint": item.split_fingerprint,
        "screening_threshold_selection_fingerprint": (
            item.screening_threshold_selection_fingerprint
        ),
    }


def _release_lane_eligible(item: DynamicPoolOutcomeEvidence) -> bool:
    return bool(
        item.human_review_decision == "include"
        and item.review_decision_fingerprint is not None
        and item.review_source_image_sha256 == item.source_image_sha256
        and item.conflict_status in {"resolved", "not_required"}
        and item.occurrence_claim_supported
        and item.release_decision.state is FlickrReleaseState.ELIGIBLE
        and item.release_decision.eligible_for_final_occurrence_dataset
        and not item.release_decision.reasons
    )


def _normalized_evidence(
    evidence: Sequence[DynamicPoolOutcomeEvidence],
) -> tuple[DynamicPoolOutcomeEvidence, ...]:
    if isinstance(evidence, str | bytes | bytearray):
        raise TypeError("outcome evidence must be a sequence")
    items = tuple(evidence)
    if any(not isinstance(item, DynamicPoolOutcomeEvidence) for item in items):
        raise TypeError("outcome evidence contains an invalid item")
    ordered = tuple(sorted(items, key=lambda item: item.item_id))
    if len({item.item_id for item in ordered}) != len(ordered):
        raise ValueError("outcome evidence item IDs must be unique")
    if len({item.source_record_id for item in ordered}) != len(ordered):
        raise ValueError("outcome evidence source record IDs must be unique")
    return ordered


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _choice(value: object, *, field: str, allowed: frozenset[str]) -> str:
    text = _required_text(value, field=field)
    if text not in allowed:
        raise ValueError(f"unsupported {field}: {text}")
    return text


def _optional_choice(
    value: object,
    *,
    field: str,
    allowed: frozenset[str],
) -> str | None:
    if value is None:
        return None
    return _choice(value, field=field, allowed=allowed)


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field).casefold()
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 fingerprint") from exc
    return text


def _optional_sha256(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field=field)


def _probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _optional_probability(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _probability(value, field=field)


__all__ = [
    "CONFLICT_STATUSES",
    "DYNAMIC_POOL_OUTCOME_EVIDENCE_VERSION",
    "HUMAN_REVIEWED_RELEASE_LABEL",
    "HUMAN_REVIEWED_RELEASE_SCHEMA",
    "HUMAN_REVIEWED_RELEASE_SCHEMA_VERSION",
    "HUMAN_REVIEW_DECISIONS",
    "SCREENING_THRESHOLD_STATUSES",
    "DynamicPoolLaneProjection",
    "DynamicPoolOutcomeEvidence",
    "project_human_reviewed_release_set",
    "validate_human_reviewed_release_set",
]
