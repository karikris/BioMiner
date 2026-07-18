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
from biominer.ml.dynamic_pool_thresholds import SCREENING_CANDIDATE_LABEL


DYNAMIC_POOL_OUTCOME_EVIDENCE_VERSION = "dynamic-pool-outcome-evidence-v1.0.0"
HUMAN_REVIEWED_RELEASE_SCHEMA_VERSION = "human-reviewed-release-lane-v1.0.0"
HUMAN_REVIEWED_RELEASE_LABEL = "human_reviewed_release_candidate"
AUDITED_SCREENING_CANDIDATE_SCHEMA_VERSION = "audited-screening-candidate-lane-v1.0.0"
UNRESOLVED_CANDIDATE_QUEUE_SCHEMA_VERSION = "unresolved-candidate-queue-lane-v1.0.0"
DYNAMIC_POOL_OUTCOME_LANES_VERSION = "dynamic-pool-outcome-lanes-v1.0.0"
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

AUDITED_SCREENING_CANDIDATE_SCHEMA: dict[str, pl.DataType] = {
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
    "review_source_image_sha256": pl.String,
    "conflict_status": pl.String,
    "human_reviewed": pl.Boolean,
    "human_review_required_before_release": pl.Boolean,
    "calibrated_supported_probability": pl.Float64,
    "screening_threshold": pl.Float64,
    "screening_threshold_status": pl.String,
    "screening_threshold_selection_fingerprint": pl.String,
    "route_compatible": pl.Boolean,
    "reference_coverage_sufficient": pl.Boolean,
    "geographic_evidence_sufficient": pl.Boolean,
    "visual_detail_sufficient": pl.Boolean,
    "domain_negative_absent": pl.Boolean,
    "out_of_distribution_absent": pl.Boolean,
    "screening_supported": pl.Boolean,
    "screening_only": pl.Boolean,
    "occurrence_claim_supported": pl.Boolean,
    "eligible_for_final_occurrence_dataset": pl.Boolean,
    "release_state": pl.String,
    "release_reasons": pl.List(pl.String),
    "release_authorized": pl.Boolean,
    "model_evidence_authorizes_release": pl.Boolean,
    "evidence_model_fingerprint": pl.String,
    "calibrator_fingerprint": pl.String,
    "split_fingerprint": pl.String,
}

UNRESOLVED_CANDIDATE_QUEUE_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "lane_fingerprint": pl.String,
    "lane_row_fingerprint": pl.String,
    "source_evidence_fingerprint": pl.String,
    "outcome_lane": pl.String,
    "outcome_label": pl.String,
    "queue_rank": pl.UInt32,
    "review_priority": pl.Float64,
    "item_id": pl.String,
    "source_record_id": pl.String,
    "candidate_species_key": pl.String,
    "route": pl.String,
    "source_image_sha256": pl.String,
    "human_review_decision": pl.String,
    "review_decision_fingerprint": pl.String,
    "review_source_image_sha256": pl.String,
    "conflict_status": pl.String,
    "human_reviewed": pl.Boolean,
    "review_required": pl.Boolean,
    "model_abstained": pl.Boolean,
    "abstention_reasons": pl.List(pl.String),
    "triage_reasons": pl.List(pl.String),
    "calibrated_supported_probability": pl.Float64,
    "screening_threshold": pl.Float64,
    "screening_threshold_status": pl.String,
    "screening_threshold_selection_fingerprint": pl.String,
    "route_compatible": pl.Boolean,
    "reference_coverage_sufficient": pl.Boolean,
    "geographic_evidence_sufficient": pl.Boolean,
    "visual_detail_sufficient": pl.Boolean,
    "domain_negative_absent": pl.Boolean,
    "out_of_distribution_absent": pl.Boolean,
    "occurrence_claim_supported": pl.Boolean,
    "eligible_for_final_occurrence_dataset": pl.Boolean,
    "release_state": pl.String,
    "release_reasons": pl.List(pl.String),
    "release_authorized": pl.Boolean,
    "model_evidence_authorizes_release": pl.Boolean,
    "evidence_model_fingerprint": pl.String,
    "calibrator_fingerprint": pl.String,
    "split_fingerprint": pl.String,
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


@dataclass(frozen=True, slots=True)
class DynamicPoolOutcomeLanes:
    human_reviewed_release: DynamicPoolLaneProjection
    audited_screening: DynamicPoolLaneProjection
    unresolved: DynamicPoolLaneProjection
    source_item_count: int
    bundle_fingerprint: str


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


def project_audited_screening_candidates(
    evidence: Sequence[DynamicPoolOutcomeEvidence],
) -> DynamicPoolLaneProjection:
    """Project unreviewed threshold-passing rows as screening candidates only."""

    items = _normalized_evidence(evidence)
    selected = tuple(item for item in items if _screening_lane_eligible(item))
    semantic_rows = [_screening_candidate_row_base(item) for item in selected]
    row_fingerprints = [canonical_semantic_fingerprint(row) for row in semantic_rows]
    lane_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": AUDITED_SCREENING_CANDIDATE_SCHEMA_VERSION,
            "source_evidence_fingerprints": [item.fingerprint for item in items],
            "projected_row_fingerprints": row_fingerprints,
            "selection_rule": "unreviewed_audited_threshold_and_quality_gates_passed",
            "outcome_label": SCREENING_CANDIDATE_LABEL,
            "occurrence_release_authorized": False,
        }
    )
    rows = [
        {
            "schema_version": AUDITED_SCREENING_CANDIDATE_SCHEMA_VERSION,
            "lane_fingerprint": lane_fingerprint,
            "lane_row_fingerprint": row_fingerprint,
            **base,
        }
        for base, row_fingerprint in zip(semantic_rows, row_fingerprints, strict=True)
    ]
    table = (
        pl.DataFrame(
            rows,
            schema=AUDITED_SCREENING_CANDIDATE_SCHEMA,
            strict=True,
        ).sort("item_id")
        if rows
        else pl.DataFrame(schema=AUDITED_SCREENING_CANDIDATE_SCHEMA)
    )
    validate_audited_screening_candidates(table)
    return DynamicPoolLaneProjection(
        table=table,
        lane_fingerprint=lane_fingerprint,
        source_item_count=len(items),
        projected_item_count=table.height,
    )


def validate_audited_screening_candidates(table: pl.DataFrame) -> None:
    """Reject language or authority drift in the unreviewed screening lane."""

    if not isinstance(table, pl.DataFrame):
        raise TypeError("screening candidate set must be a Polars DataFrame")
    if table.schema != AUDITED_SCREENING_CANDIDATE_SCHEMA:
        raise ValueError("screening candidate schema does not match contract")
    if not table.height:
        return
    if not table.equals(table.sort("item_id")):
        raise ValueError("screening candidate set is not sorted")
    if table["item_id"].n_unique() != table.height:
        raise ValueError("screening candidate item IDs must be unique")
    if table.filter(
        (pl.col("schema_version") != AUDITED_SCREENING_CANDIDATE_SCHEMA_VERSION)
        | (pl.col("outcome_lane") != "statistical_screening")
        | (pl.col("outcome_label") != SCREENING_CANDIDATE_LABEL)
        | pl.col("human_review_decision").is_not_null()
        | pl.col("review_source_image_sha256").is_not_null()
        | pl.col("human_reviewed")
        | ~pl.col("human_review_required_before_release")
        | (pl.col("screening_threshold_status") != "selected")
        | (pl.col("calibrated_supported_probability") < pl.col("screening_threshold"))
        | ~pl.col("route_compatible")
        | ~pl.col("reference_coverage_sufficient")
        | ~pl.col("geographic_evidence_sufficient")
        | ~pl.col("visual_detail_sufficient")
        | ~pl.col("domain_negative_absent")
        | ~pl.col("out_of_distribution_absent")
        | ~pl.col("screening_supported")
        | ~pl.col("screening_only")
        | pl.col("occurrence_claim_supported")
        | pl.col("eligible_for_final_occurrence_dataset")
        | (pl.col("release_state") != "excluded")
        | pl.col("release_authorized")
        | pl.col("model_evidence_authorizes_release")
    ).height:
        raise ValueError("screening candidate crossed its evidence boundary")
    for row in table.iter_rows(named=True):
        base = {
            field: row[field]
            for field in AUDITED_SCREENING_CANDIDATE_SCHEMA
            if field
            not in {"schema_version", "lane_fingerprint", "lane_row_fingerprint"}
        }
        if row["lane_row_fingerprint"] != canonical_semantic_fingerprint(base):
            raise ValueError("screening candidate row fingerprint mismatch")
    if table["lane_fingerprint"].n_unique() != 1:
        raise ValueError("screening candidate set has mixed lane fingerprints")


def project_unresolved_candidate_queue(
    evidence: Sequence[DynamicPoolOutcomeEvidence],
) -> DynamicPoolLaneProjection:
    """Retain every candidate in neither release nor audited screening."""

    items = _normalized_evidence(evidence)
    selected = tuple(
        item
        for item in items
        if not _release_lane_eligible(item) and not _screening_lane_eligible(item)
    )
    ordered = tuple(
        sorted(selected, key=lambda item: (-item.review_priority, item.item_id))
    )
    semantic_rows = [
        _unresolved_row_base(item, queue_rank=index)
        for index, item in enumerate(ordered, start=1)
    ]
    row_fingerprints = [canonical_semantic_fingerprint(row) for row in semantic_rows]
    lane_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": UNRESOLVED_CANDIDATE_QUEUE_SCHEMA_VERSION,
            "source_evidence_fingerprints": [item.fingerprint for item in items],
            "projected_row_fingerprints": row_fingerprints,
            "selection_rule": "not_human_release_and_not_audited_screening",
            "occurrence_release_authorized": False,
        }
    )
    rows = [
        {
            "schema_version": UNRESOLVED_CANDIDATE_QUEUE_SCHEMA_VERSION,
            "lane_fingerprint": lane_fingerprint,
            "lane_row_fingerprint": row_fingerprint,
            **base,
        }
        for base, row_fingerprint in zip(semantic_rows, row_fingerprints, strict=True)
    ]
    table = (
        pl.DataFrame(
            rows,
            schema=UNRESOLVED_CANDIDATE_QUEUE_SCHEMA,
            strict=True,
        )
        if rows
        else pl.DataFrame(schema=UNRESOLVED_CANDIDATE_QUEUE_SCHEMA)
    )
    validate_unresolved_candidate_queue(table)
    return DynamicPoolLaneProjection(
        table=table,
        lane_fingerprint=lane_fingerprint,
        source_item_count=len(items),
        projected_item_count=table.height,
    )


def validate_unresolved_candidate_queue(table: pl.DataFrame) -> None:
    """Verify deterministic ranking and fail-closed unresolved authority."""

    if not isinstance(table, pl.DataFrame):
        raise TypeError("unresolved candidate queue must be a Polars DataFrame")
    if table.schema != UNRESOLVED_CANDIDATE_QUEUE_SCHEMA:
        raise ValueError("unresolved candidate queue schema does not match contract")
    if not table.height:
        return
    if table["item_id"].n_unique() != table.height:
        raise ValueError("unresolved candidate item IDs must be unique")
    if table["queue_rank"].to_list() != list(range(1, table.height + 1)):
        raise ValueError("unresolved candidate queue ranks must be contiguous")
    expected_order = sorted(
        table.to_dicts(),
        key=lambda row: (-float(row["review_priority"]), str(row["item_id"])),
    )
    if table["item_id"].to_list() != [row["item_id"] for row in expected_order]:
        raise ValueError("unresolved candidate queue order is invalid")
    if table.filter(
        (pl.col("schema_version") != UNRESOLVED_CANDIDATE_QUEUE_SCHEMA_VERSION)
        | (pl.col("outcome_lane") != "review_required_or_abstained")
        | ~pl.col("outcome_label").is_in(["review_required", "human_review_excluded"])
        | ~pl.col("model_abstained")
        | (pl.col("abstention_reasons").list.len() == 0)
        | pl.col("eligible_for_final_occurrence_dataset")
        | (pl.col("release_state") != "excluded")
        | pl.col("release_authorized")
        | pl.col("model_evidence_authorizes_release")
    ).height:
        raise ValueError("unresolved candidate crossed its evidence boundary")
    if table.filter(
        (pl.col("outcome_label") == "human_review_excluded") & pl.col("review_required")
    ).height:
        raise ValueError("human review exclusion cannot request identity confirmation")
    if table.filter(
        (pl.col("outcome_label") == "review_required") & ~pl.col("review_required")
    ).height:
        raise ValueError("review-required outcome must remain reviewable")
    for row in table.iter_rows(named=True):
        base = {
            field: row[field]
            for field in UNRESOLVED_CANDIDATE_QUEUE_SCHEMA
            if field
            not in {"schema_version", "lane_fingerprint", "lane_row_fingerprint"}
        }
        if row["lane_row_fingerprint"] != canonical_semantic_fingerprint(base):
            raise ValueError("unresolved candidate row fingerprint mismatch")
    if table["lane_fingerprint"].n_unique() != 1:
        raise ValueError("unresolved candidate queue has mixed lane fingerprints")


def project_dynamic_pool_outcome_lanes(
    evidence: Sequence[DynamicPoolOutcomeEvidence],
) -> DynamicPoolOutcomeLanes:
    """Project a complete and mutually exclusive three-lane outcome partition."""

    items = _normalized_evidence(evidence)
    release = project_human_reviewed_release_set(items)
    screening = project_audited_screening_candidates(items)
    unresolved = project_unresolved_candidate_queue(items)
    source_ids = {item.item_id for item in items}
    lane_sets = (
        set(release.table["item_id"].to_list()),
        set(screening.table["item_id"].to_list()),
        set(unresolved.table["item_id"].to_list()),
    )
    if any(
        lane_sets[left] & lane_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise AssertionError("dynamic-pool outcome lanes overlap")
    if set().union(*lane_sets) != source_ids:
        raise AssertionError("dynamic-pool outcome lanes do not cover source evidence")
    bundle_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_OUTCOME_LANES_VERSION,
            "source_evidence_fingerprints": [item.fingerprint for item in items],
            "human_reviewed_release_fingerprint": release.lane_fingerprint,
            "audited_screening_fingerprint": screening.lane_fingerprint,
            "unresolved_fingerprint": unresolved.lane_fingerprint,
            "lane_item_ids": [sorted(values) for values in lane_sets],
            "complete": True,
            "mutually_exclusive": True,
            "unreviewed_occurrence_export_count": 0,
        }
    )
    return DynamicPoolOutcomeLanes(
        human_reviewed_release=release,
        audited_screening=screening,
        unresolved=unresolved,
        source_item_count=len(items),
        bundle_fingerprint=bundle_fingerprint,
    )


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


def _screening_candidate_row_base(
    item: DynamicPoolOutcomeEvidence,
) -> dict[str, object]:
    assert item.human_review_decision is None
    assert item.calibrated_supported_probability is not None
    assert item.screening_threshold is not None
    assert item.screening_threshold_selection_fingerprint is not None
    return {
        "source_evidence_fingerprint": item.fingerprint,
        "outcome_lane": "statistical_screening",
        "outcome_label": SCREENING_CANDIDATE_LABEL,
        "item_id": item.item_id,
        "source_record_id": item.source_record_id,
        "candidate_species_key": item.candidate_species_key,
        "route": item.route,
        "source_image_sha256": item.source_image_sha256,
        "human_review_decision": None,
        "review_source_image_sha256": None,
        "conflict_status": item.conflict_status,
        "human_reviewed": False,
        "human_review_required_before_release": True,
        "calibrated_supported_probability": item.calibrated_supported_probability,
        "screening_threshold": item.screening_threshold,
        "screening_threshold_status": item.screening_threshold_status,
        "screening_threshold_selection_fingerprint": (
            item.screening_threshold_selection_fingerprint
        ),
        "route_compatible": item.route_compatible,
        "reference_coverage_sufficient": item.reference_coverage_sufficient,
        "geographic_evidence_sufficient": item.geographic_evidence_sufficient,
        "visual_detail_sufficient": item.visual_detail_sufficient,
        "domain_negative_absent": item.domain_negative_absent,
        "out_of_distribution_absent": item.out_of_distribution_absent,
        "screening_supported": True,
        "screening_only": True,
        "occurrence_claim_supported": False,
        "eligible_for_final_occurrence_dataset": False,
        "release_state": "excluded",
        "release_reasons": [
            "human_review_missing",
            "screening_only_not_occurrence_release",
        ],
        "release_authorized": False,
        "model_evidence_authorizes_release": False,
        "evidence_model_fingerprint": item.evidence_model_fingerprint,
        "calibrator_fingerprint": item.calibrator_fingerprint,
        "split_fingerprint": item.split_fingerprint,
    }


def _unresolved_row_base(
    item: DynamicPoolOutcomeEvidence,
    *,
    queue_rank: int,
) -> dict[str, object]:
    reasons = _abstention_reasons(item)
    human_excluded = item.human_review_decision == "exclude"
    release_reasons = [str(reason) for reason in item.release_decision.reasons]
    if "not_release_lane_eligible" not in release_reasons:
        release_reasons.append("not_release_lane_eligible")
    return {
        "source_evidence_fingerprint": item.fingerprint,
        "outcome_lane": "review_required_or_abstained",
        "outcome_label": (
            "human_review_excluded" if human_excluded else "review_required"
        ),
        "queue_rank": queue_rank,
        "review_priority": item.review_priority,
        "item_id": item.item_id,
        "source_record_id": item.source_record_id,
        "candidate_species_key": item.candidate_species_key,
        "route": item.route,
        "source_image_sha256": item.source_image_sha256,
        "human_review_decision": item.human_review_decision,
        "review_decision_fingerprint": item.review_decision_fingerprint,
        "review_source_image_sha256": item.review_source_image_sha256,
        "conflict_status": item.conflict_status,
        "human_reviewed": item.human_review_decision is not None,
        "review_required": not human_excluded,
        "model_abstained": True,
        "abstention_reasons": reasons,
        "triage_reasons": list(item.triage_reasons),
        "calibrated_supported_probability": item.calibrated_supported_probability,
        "screening_threshold": item.screening_threshold,
        "screening_threshold_status": item.screening_threshold_status,
        "screening_threshold_selection_fingerprint": (
            item.screening_threshold_selection_fingerprint
        ),
        "route_compatible": item.route_compatible,
        "reference_coverage_sufficient": item.reference_coverage_sufficient,
        "geographic_evidence_sufficient": item.geographic_evidence_sufficient,
        "visual_detail_sufficient": item.visual_detail_sufficient,
        "domain_negative_absent": item.domain_negative_absent,
        "out_of_distribution_absent": item.out_of_distribution_absent,
        "occurrence_claim_supported": item.occurrence_claim_supported,
        "eligible_for_final_occurrence_dataset": False,
        "release_state": "excluded",
        "release_reasons": release_reasons,
        "release_authorized": False,
        "model_evidence_authorizes_release": False,
        "evidence_model_fingerprint": item.evidence_model_fingerprint,
        "calibrator_fingerprint": item.calibrator_fingerprint,
        "split_fingerprint": item.split_fingerprint,
    }


def _abstention_reasons(item: DynamicPoolOutcomeEvidence) -> list[str]:
    reasons = list(item.triage_reasons)
    if item.human_review_decision is None:
        reasons.append("human_review_missing")
    elif item.human_review_decision == "uncertain":
        reasons.append("human_review_not_decisive")
    elif item.human_review_decision == "exclude":
        reasons.append("human_review_excluded")
    if item.conflict_status not in {"resolved", "not_required"}:
        reasons.append("review_conflict_unresolved")
    if item.screening_threshold_status != "selected":
        reasons.append(f"screening_threshold_{item.screening_threshold_status}")
    elif item.calibrated_supported_probability is None:
        reasons.append("calibrated_probability_missing")
    elif (
        item.screening_threshold is not None
        and item.calibrated_supported_probability < item.screening_threshold
    ):
        reasons.append("calibrated_probability_below_screening_threshold")
    for passed, reason in (
        (item.route_compatible, "route_incompatible"),
        (item.reference_coverage_sufficient, "reference_coverage_insufficient"),
        (item.geographic_evidence_sufficient, "geographic_evidence_insufficient"),
        (item.visual_detail_sufficient, "visual_detail_insufficient"),
        (item.domain_negative_absent, "domain_negative_detected"),
        (item.out_of_distribution_absent, "out_of_distribution"),
    ):
        if not passed:
            reasons.append(reason)
    reasons.extend(str(reason) for reason in item.release_decision.reasons)
    if not reasons:
        reasons.append("not_release_or_screening_eligible")
    return list(dict.fromkeys(reasons))


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


def _screening_lane_eligible(item: DynamicPoolOutcomeEvidence) -> bool:
    return bool(
        item.human_review_decision is None
        and item.review_decision_fingerprint is None
        and item.review_source_image_sha256 is None
        and item.screening_threshold_status == "selected"
        and item.screening_threshold_selection_fingerprint is not None
        and item.screening_threshold is not None
        and item.calibrated_supported_probability is not None
        and item.calibrated_supported_probability >= item.screening_threshold
        and item.route_compatible
        and item.reference_coverage_sufficient
        and item.geographic_evidence_sufficient
        and item.visual_detail_sufficient
        and item.domain_negative_absent
        and item.out_of_distribution_absent
        and not item.occurrence_claim_supported
        and item.release_decision.state is FlickrReleaseState.EXCLUDED
        and not item.release_decision.eligible_for_final_occurrence_dataset
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
    "AUDITED_SCREENING_CANDIDATE_SCHEMA",
    "AUDITED_SCREENING_CANDIDATE_SCHEMA_VERSION",
    "CONFLICT_STATUSES",
    "DYNAMIC_POOL_OUTCOME_EVIDENCE_VERSION",
    "DYNAMIC_POOL_OUTCOME_LANES_VERSION",
    "HUMAN_REVIEWED_RELEASE_LABEL",
    "HUMAN_REVIEWED_RELEASE_SCHEMA",
    "HUMAN_REVIEWED_RELEASE_SCHEMA_VERSION",
    "HUMAN_REVIEW_DECISIONS",
    "SCREENING_THRESHOLD_STATUSES",
    "UNRESOLVED_CANDIDATE_QUEUE_SCHEMA",
    "UNRESOLVED_CANDIDATE_QUEUE_SCHEMA_VERSION",
    "DynamicPoolLaneProjection",
    "DynamicPoolOutcomeEvidence",
    "DynamicPoolOutcomeLanes",
    "project_audited_screening_candidates",
    "project_dynamic_pool_outcome_lanes",
    "project_human_reviewed_release_set",
    "project_unresolved_candidate_queue",
    "validate_audited_screening_candidates",
    "validate_human_reviewed_release_set",
    "validate_unresolved_candidate_queue",
]
