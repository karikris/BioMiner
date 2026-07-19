"""Additional Flickr audit work projected from dynamic-pool escalations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.dynamic_pool_escalation import validate_pooling_escalations


DYNAMIC_POOL_FLICKR_AUDIT_CANDIDATE_VERSION = (
    "dynamic-pool-flickr-audit-candidate-v1.0.0"
)
DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_VERSION = (
    "dynamic-pool-flickr-audit-expansion-v1.0.0"
)
DYNAMIC_POOL_FLICKR_AUDIT_PROJECTION_VERSION = (
    "dynamic-pool-flickr-audit-projection-v1.0.0"
)
REPRESENTATIVE_EXPANSION_ACTION = "collect_additional_representative_flickr_reviews"

AUDIT_PRIORITY_COMPONENT_SCHEMA = pl.Struct(
    {
        "component": pl.String,
        "observed": pl.Float64,
        "normalized_score": pl.Float64,
    }
)

DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "queue_fingerprint": pl.String,
    "queue_row_fingerprint": pl.String,
    "candidate_fingerprint": pl.String,
    "priority_rank": pl.UInt32,
    "priority_score": pl.Float64,
    "priority_score_semantics": pl.String,
    "priority_components": pl.List(AUDIT_PRIORITY_COMPONENT_SCHEMA),
    "queue_kinds": pl.List(pl.String),
    "item_id": pl.String,
    "source_record_id": pl.String,
    "source": pl.String,
    "source_image_sha256": pl.String,
    "audit_unit_fingerprint": pl.String,
    "independence_component_id": pl.String,
    "family_key": pl.String,
    "family_name": pl.String,
    "genus_key": pl.String,
    "genus_name": pl.String,
    "species_key": pl.String,
    "scientific_name": pl.String,
    "route": pl.String,
    "country_code": pl.String,
    "admin1": pl.String,
    "bioregion": pl.String,
    "geographic_cluster_id": pl.String,
    "no_geo": pl.Boolean,
    "matched_escalation_group_ids": pl.List(pl.String),
    "matched_escalation_decision_fingerprints": pl.List(pl.String),
    "trigger_reasons": pl.List(pl.String),
    "recommended_actions": pl.List(pl.String),
    "sampling_design_status": pl.String,
    "sampling_purpose": pl.String,
    "inclusion_probability": pl.Float64,
    "sampling_weight": pl.Float64,
    "representative_estimation_eligible": pl.Boolean,
    "review_status": pl.String,
    "human_review_required": pl.Boolean,
    "occurrence_claim_supported": pl.Boolean,
    "eligible_for_final_occurrence_dataset": pl.Boolean,
    "release_authorized": pl.Boolean,
}


@dataclass(frozen=True, slots=True)
class DynamicPoolFlickrAuditCandidate:
    """One unreviewed Flickr unit eligible for additional audit planning."""

    item_id: str
    source_record_id: str
    source_image_sha256: str
    audit_unit_fingerprint: str
    independence_component_id: str
    family_key: str
    family_name: str
    genus_key: str
    genus_name: str
    species_key: str
    scientific_name: str
    route: str
    review_priority: float
    global_local_disagreement: float
    local_support_available: bool
    out_of_distribution: bool
    calibrated_supported_probability: float | None = None
    country_code: str | None = None
    admin1: str | None = None
    bioregion: str | None = None
    geographic_cluster_id: str | None = None
    no_geo: bool = False
    source: str = "flickr"
    schema_version: str = DYNAMIC_POOL_FLICKR_AUDIT_CANDIDATE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_POOL_FLICKR_AUDIT_CANDIDATE_VERSION:
            raise ValueError("unsupported dynamic-pool Flickr audit candidate version")
        for field in (
            "item_id",
            "source_record_id",
            "independence_component_id",
            "family_key",
            "family_name",
            "genus_key",
            "genus_name",
            "species_key",
            "scientific_name",
            "route",
        ):
            object.__setattr__(
                self, field, _required_text(getattr(self, field), field=field)
            )
        source = _required_text(self.source, field="source").casefold()
        if source != "flickr":
            raise ValueError("dynamic-pool audit expansion requires Flickr source")
        object.__setattr__(self, "source", source)
        for field in ("source_image_sha256", "audit_unit_fingerprint"):
            object.__setattr__(self, field, _sha256(getattr(self, field), field=field))
        object.__setattr__(
            self,
            "review_priority",
            _probability(self.review_priority, field="review_priority"),
        )
        disagreement = _finite_float(
            self.global_local_disagreement,
            field="global_local_disagreement",
        )
        if not 0.0 <= disagreement <= 2.0:
            raise ValueError("global_local_disagreement must be within [0, 2]")
        object.__setattr__(self, "global_local_disagreement", disagreement)
        for field in ("local_support_available", "out_of_distribution", "no_geo"):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be a boolean")
        if self.calibrated_supported_probability is not None:
            object.__setattr__(
                self,
                "calibrated_supported_probability",
                _probability(
                    self.calibrated_supported_probability,
                    field="calibrated_supported_probability",
                ),
            )
        for field in (
            "country_code",
            "admin1",
            "bioregion",
            "geographic_cluster_id",
        ):
            object.__setattr__(
                self,
                field,
                _optional_text(getattr(self, field), field=field),
            )
        if self.country_code is not None:
            object.__setattr__(self, "country_code", self.country_code.upper())
        if not self.no_geo and self.geographic_cluster_id is None:
            raise ValueError(
                "geocoded Flickr audit candidate requires geographic cluster"
            )

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class DynamicPoolFlickrAuditProjection:
    queue: pl.DataFrame
    projection_fingerprint: str
    source_candidate_count: int
    queued_candidate_count: int
    matched_escalation_group_ids: tuple[str, ...]
    unmatched_escalation_group_ids: tuple[str, ...]


def build_additional_flickr_audit_queue(
    candidates: Sequence[DynamicPoolFlickrAuditCandidate],
    escalations: pl.DataFrame,
) -> DynamicPoolFlickrAuditProjection:
    """Project follow-up work without assigning an invented sampling design."""

    validate_pooling_escalations(escalations)
    items = _normalized_candidates(candidates)
    flagged = escalations.filter(
        pl.col("flagged_for_remediation") & pl.col("additional_flickr_audit_candidate")
    ).sort(["hierarchy_level", "group_id"])
    matched_group_ids: set[str] = set()
    semantic_rows: list[dict[str, object]] = []
    for item in items:
        matches = [
            row
            for row in flagged.iter_rows(named=True)
            if _candidate_matches_escalation(item, row)
        ]
        if not matches:
            continue
        group_ids = sorted({str(row["group_id"]) for row in matches})
        matched_group_ids.update(group_ids)
        actions = sorted(
            {str(action) for row in matches for action in row["recommended_actions"]}
        )
        queue_kinds = []
        if REPRESENTATIVE_EXPANSION_ACTION in actions:
            queue_kinds.append("representative_audit_expansion_candidate")
        if set(actions) - {REPRESENTATIVE_EXPANSION_ACTION}:
            queue_kinds.append("targeted_diagnostic_followup")
        requires_design = "representative_audit_expansion_candidate" in queue_kinds
        components = _priority_components(item)
        score = sum(
            float(component["normalized_score"]) for component in components
        ) / len(components)
        semantic_rows.append(
            {
                "candidate_fingerprint": item.fingerprint,
                "priority_rank": 0,
                "priority_score": score,
                "priority_score_semantics": "transparent_review_heuristic_not_probability",
                "priority_components": components,
                "queue_kinds": queue_kinds,
                "item_id": item.item_id,
                "source_record_id": item.source_record_id,
                "source": item.source,
                "source_image_sha256": item.source_image_sha256,
                "audit_unit_fingerprint": item.audit_unit_fingerprint,
                "independence_component_id": item.independence_component_id,
                "family_key": item.family_key,
                "family_name": item.family_name,
                "genus_key": item.genus_key,
                "genus_name": item.genus_name,
                "species_key": item.species_key,
                "scientific_name": item.scientific_name,
                "route": item.route,
                "country_code": item.country_code,
                "admin1": item.admin1,
                "bioregion": item.bioregion,
                "geographic_cluster_id": item.geographic_cluster_id,
                "no_geo": item.no_geo,
                "matched_escalation_group_ids": group_ids,
                "matched_escalation_decision_fingerprints": sorted(
                    {str(row["decision_fingerprint"]) for row in matches}
                ),
                "trigger_reasons": sorted(
                    {
                        str(reason)
                        for row in matches
                        for reason in row["trigger_reasons"]
                    }
                ),
                "recommended_actions": actions,
                "sampling_design_status": (
                    "required_before_representative_estimation"
                    if requires_design
                    else "not_applicable_targeted_followup"
                ),
                "sampling_purpose": (
                    "pending_probability_design"
                    if requires_design
                    else "targeted_failure_discovery"
                ),
                "inclusion_probability": None,
                "sampling_weight": None,
                "representative_estimation_eligible": False,
                "review_status": "pending",
                "human_review_required": True,
                "occurrence_claim_supported": False,
                "eligible_for_final_occurrence_dataset": False,
                "release_authorized": False,
            }
        )
    semantic_rows.sort(
        key=lambda row: (-float(row["priority_score"]), str(row["item_id"]))
    )
    for rank, row in enumerate(semantic_rows, start=1):
        row["priority_rank"] = rank
    row_fingerprints = [canonical_semantic_fingerprint(row) for row in semantic_rows]
    flagged_ids = set(flagged["group_id"].to_list())
    unmatched_group_ids = tuple(sorted(flagged_ids - matched_group_ids))
    queue_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_VERSION,
            "escalation_report_fingerprint": escalations["report_fingerprint"].item(0),
            "source_candidate_fingerprints": [item.fingerprint for item in items],
            "queue_row_fingerprints": row_fingerprints,
            "matched_escalation_group_ids": sorted(matched_group_ids),
            "unmatched_escalation_group_ids": unmatched_group_ids,
        }
    )
    rows = [
        {
            "schema_version": DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_VERSION,
            "queue_fingerprint": queue_fingerprint,
            "queue_row_fingerprint": row_fingerprint,
            **row,
        }
        for row, row_fingerprint in zip(semantic_rows, row_fingerprints, strict=True)
    ]
    queue = (
        pl.DataFrame(
            rows,
            schema=DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_SCHEMA,
            strict=True,
        )
        if rows
        else pl.DataFrame(schema=DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_SCHEMA)
    )
    validate_additional_flickr_audit_queue(queue)
    projection_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_FLICKR_AUDIT_PROJECTION_VERSION,
            "queue_fingerprint": queue_fingerprint,
            "source_candidate_count": len(items),
            "queued_candidate_count": queue.height,
            "matched_escalation_group_ids": sorted(matched_group_ids),
            "unmatched_escalation_group_ids": unmatched_group_ids,
        }
    )
    return DynamicPoolFlickrAuditProjection(
        queue=queue,
        projection_fingerprint=projection_fingerprint,
        source_candidate_count=len(items),
        queued_candidate_count=queue.height,
        matched_escalation_group_ids=tuple(sorted(matched_group_ids)),
        unmatched_escalation_group_ids=unmatched_group_ids,
    )


def validate_additional_flickr_audit_queue(queue: pl.DataFrame) -> None:
    """Keep follow-up audit work outside estimation and occurrence release."""

    if not isinstance(queue, pl.DataFrame):
        raise TypeError("additional Flickr audit queue must be a Polars DataFrame")
    if queue.schema != DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_SCHEMA:
        raise ValueError("additional Flickr audit queue schema does not match contract")
    if queue.is_empty():
        return
    if queue["item_id"].n_unique() != queue.height:
        raise ValueError("additional Flickr audit queue repeats an item ID")
    if queue["source_record_id"].n_unique() != queue.height:
        raise ValueError("additional Flickr audit queue repeats a source record")
    if queue["priority_rank"].to_list() != list(range(1, queue.height + 1)):
        raise ValueError("additional Flickr audit ranks must be contiguous")
    if not queue.equals(
        queue.sort(["priority_score", "item_id"], descending=[True, False])
    ):
        raise ValueError("additional Flickr audit queue is not canonically sorted")
    if queue.filter(
        (pl.col("schema_version") != DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_VERSION)
        | (pl.col("source") != "flickr")
        | (
            pl.col("priority_score_semantics")
            != "transparent_review_heuristic_not_probability"
        )
        | (pl.col("queue_kinds").list.len() == 0)
        | pl.col("inclusion_probability").is_not_null()
        | pl.col("sampling_weight").is_not_null()
        | pl.col("representative_estimation_eligible")
        | (pl.col("review_status") != "pending")
        | ~pl.col("human_review_required")
        | pl.col("occurrence_claim_supported")
        | pl.col("eligible_for_final_occurrence_dataset")
        | pl.col("release_authorized")
    ).height:
        raise ValueError("additional Flickr audit queue crossed its evidence boundary")
    representative = pl.col("queue_kinds").list.contains(
        "representative_audit_expansion_candidate"
    )
    if queue.filter(
        representative
        & (
            (
                pl.col("sampling_design_status")
                != "required_before_representative_estimation"
            )
            | (pl.col("sampling_purpose") != "pending_probability_design")
        )
    ).height:
        raise ValueError("representative audit expansion lacks a sampling-design gate")
    for row in queue.iter_rows(named=True):
        base = {
            field: row[field]
            for field in DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_SCHEMA
            if field
            not in {"schema_version", "queue_fingerprint", "queue_row_fingerprint"}
        }
        if row["queue_row_fingerprint"] != canonical_semantic_fingerprint(base):
            raise ValueError("additional Flickr audit row fingerprint mismatch")
    if queue["queue_fingerprint"].n_unique() != 1:
        raise ValueError("additional Flickr audit queue has mixed fingerprints")


def _candidate_matches_escalation(
    candidate: DynamicPoolFlickrAuditCandidate,
    escalation: dict[str, object],
) -> bool:
    hierarchy = str(escalation["hierarchy_level"])
    if hierarchy == "overall":
        return True
    if hierarchy == "family":
        return candidate.family_key == escalation["family_key"]
    if hierarchy == "genus":
        return candidate.genus_key == escalation["genus_key"]
    if hierarchy == "species":
        return candidate.species_key == escalation["species_key"]
    if hierarchy != "geography":
        return False
    level = str(escalation["geography_level"])
    value = str(escalation["geography_value"])
    observed = {
        "availability": "no_geo" if candidate.no_geo else "geocoded",
        "country": candidate.country_code or "unknown_country",
        "admin1": candidate.admin1 or "unknown_admin1",
        "bioregion": candidate.bioregion or "unknown_bioregion",
        "geographic_cluster": (
            "no_geo" if candidate.no_geo else candidate.geographic_cluster_id
        ),
    }.get(level)
    return observed == value


def _priority_components(
    candidate: DynamicPoolFlickrAuditCandidate,
) -> list[dict[str, object]]:
    probability = candidate.calibrated_supported_probability
    uncertainty = 1.0 if probability is None else 1.0 - abs(probability - 0.5) * 2.0
    values = (
        ("configured_review_priority", candidate.review_priority),
        ("global_local_disagreement", candidate.global_local_disagreement / 2.0),
        ("calibrated_probability_uncertainty", uncertainty),
        ("local_support_unavailable", float(not candidate.local_support_available)),
        ("out_of_distribution", float(candidate.out_of_distribution)),
    )
    return [
        {
            "component": component,
            "observed": float(value),
            "normalized_score": float(value),
        }
        for component, value in values
    ]


def _normalized_candidates(
    candidates: Sequence[DynamicPoolFlickrAuditCandidate],
) -> tuple[DynamicPoolFlickrAuditCandidate, ...]:
    items = tuple(candidates)
    if any(not isinstance(item, DynamicPoolFlickrAuditCandidate) for item in items):
        raise TypeError("Flickr audit candidates must use the expansion contract")
    ordered = tuple(sorted(items, key=lambda item: item.item_id))
    if len({item.item_id for item in ordered}) != len(ordered):
        raise ValueError("Flickr audit candidate item IDs must be unique")
    if len({item.source_record_id for item in ordered}) != len(ordered):
        raise ValueError("Flickr audit candidate source records must be unique")
    return ordered


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field).casefold()
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 fingerprint") from exc
    return text


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _probability(value: object, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return result


__all__ = [
    "AUDIT_PRIORITY_COMPONENT_SCHEMA",
    "DYNAMIC_POOL_FLICKR_AUDIT_CANDIDATE_VERSION",
    "DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_SCHEMA",
    "DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_VERSION",
    "DYNAMIC_POOL_FLICKR_AUDIT_PROJECTION_VERSION",
    "REPRESENTATIVE_EXPANSION_ACTION",
    "DynamicPoolFlickrAuditCandidate",
    "DynamicPoolFlickrAuditProjection",
    "build_additional_flickr_audit_queue",
    "validate_additional_flickr_audit_queue",
]
