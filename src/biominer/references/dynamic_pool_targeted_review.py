"""Source-bound GBIF reference review from dynamic-pool escalation evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.dynamic_pool_escalation import validate_pooling_escalations


DYNAMIC_POOL_REFERENCE_REVIEW_CANDIDATE_VERSION = (
    "dynamic-pool-reference-review-candidate-v1.0.0"
)
DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_VERSION = (
    "dynamic-pool-targeted-reference-queue-v1.0.0"
)
DYNAMIC_POOL_TARGETED_REFERENCE_PROJECTION_VERSION = (
    "dynamic-pool-targeted-reference-projection-v1.0.0"
)

REFERENCE_PRIORITY_COMPONENT_SCHEMA = pl.Struct(
    {
        "component": pl.String,
        "observed": pl.Float64,
        "normalized_score": pl.Float64,
    }
)

DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "queue_fingerprint": pl.String,
    "queue_row_fingerprint": pl.String,
    "candidate_fingerprint": pl.String,
    "priority_rank": pl.UInt32,
    "priority_score": pl.Float64,
    "priority_score_semantics": pl.String,
    "priority_components": pl.List(REFERENCE_PRIORITY_COMPONENT_SCHEMA),
    "reference_media_id": pl.String,
    "reference_observation_id": pl.String,
    "source": pl.String,
    "source_dataset_key": pl.String,
    "source_media_sha256": pl.String,
    "reference_bank_version": pl.String,
    "admission_policy_fingerprint": pl.String,
    "embedding_fingerprint": pl.String,
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
    "reference_quality_flags": pl.List(pl.String),
    "matched_escalation_group_ids": pl.List(pl.String),
    "matched_escalation_decision_fingerprints": pl.List(pl.String),
    "trigger_reasons": pl.List(pl.String),
    "recommended_actions": pl.List(pl.String),
    "reference_identity_conclusion": pl.String,
    "review_status": pl.String,
    "human_review_required": pl.Boolean,
    "automatic_reference_exclusion": pl.Boolean,
    "current_support_disposition": pl.String,
    "support_disposition_after_targeting": pl.String,
    "authorizes_occurrence_release": pl.Boolean,
}


@dataclass(frozen=True, slots=True)
class DynamicPoolReferenceReviewCandidate:
    """One admitted GBIF reference plus transparent review-priority evidence."""

    reference_media_id: str
    reference_observation_id: str
    source_dataset_key: str
    source_media_sha256: str
    reference_bank_version: str
    admission_policy_fingerprint: str
    embedding_fingerprint: str
    family_key: str
    family_name: str
    genus_key: str
    genus_name: str
    species_key: str
    scientific_name: str
    route: str
    embedding_outlier_score: float
    prototype_influence: float
    repeated_error_involvement_count: int
    route_domain_mismatch: bool
    local_scope_member: bool
    current_support_disposition: str
    country_code: str | None = None
    admin1: str | None = None
    bioregion: str | None = None
    geographic_cluster_id: str | None = None
    no_geo: bool = False
    reference_quality_flags: tuple[str, ...] = ()
    source: str = "gbif"
    schema_version: str = DYNAMIC_POOL_REFERENCE_REVIEW_CANDIDATE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_POOL_REFERENCE_REVIEW_CANDIDATE_VERSION:
            raise ValueError("unsupported dynamic-pool reference candidate version")
        for field in (
            "reference_media_id",
            "reference_observation_id",
            "source_dataset_key",
            "reference_bank_version",
            "family_key",
            "family_name",
            "genus_key",
            "genus_name",
            "species_key",
            "scientific_name",
            "route",
            "current_support_disposition",
        ):
            object.__setattr__(
                self, field, _required_text(getattr(self, field), field=field)
            )
        source = _required_text(self.source, field="source").casefold()
        if source != "gbif":
            raise ValueError("dynamic-pool targeted references must have GBIF source")
        object.__setattr__(self, "source", source)
        for field in (
            "source_media_sha256",
            "admission_policy_fingerprint",
            "embedding_fingerprint",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field=field))
        for field in ("embedding_outlier_score", "prototype_influence"):
            object.__setattr__(
                self,
                field,
                _probability(getattr(self, field), field=field),
            )
        count = self.repeated_error_involvement_count
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("repeated_error_involvement_count must be nonnegative")
        for field in ("route_domain_mismatch", "local_scope_member", "no_geo"):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be a boolean")
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
            raise ValueError("geocoded GBIF reference requires geographic_cluster_id")
        flags = tuple(
            sorted(
                {
                    _required_text(value, field="reference_quality_flags")
                    for value in self.reference_quality_flags
                }
            )
        )
        object.__setattr__(self, "reference_quality_flags", flags)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class DynamicPoolTargetedReferenceProjection:
    queue: pl.DataFrame
    projection_fingerprint: str
    source_candidate_count: int
    targeted_candidate_count: int
    matched_escalation_group_ids: tuple[str, ...]
    unmatched_escalation_group_ids: tuple[str, ...]


def build_dynamic_pool_targeted_reference_review_queue(
    candidates: Sequence[DynamicPoolReferenceReviewCandidate],
    escalations: pl.DataFrame,
) -> DynamicPoolTargetedReferenceProjection:
    """Target only GBIF references safely bound to review-candidate triggers."""

    validate_pooling_escalations(escalations)
    items = _normalized_candidates(candidates)
    flagged = escalations.filter(
        pl.col("flagged_for_remediation") & pl.col("reference_review_candidate")
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
        decision_fingerprints = sorted(
            {str(row["decision_fingerprint"]) for row in matches}
        )
        reasons = sorted(
            {str(reason) for row in matches for reason in row["trigger_reasons"]}
        )
        actions = sorted(
            {str(action) for row in matches for action in row["recommended_actions"]}
        )
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
                "reference_media_id": item.reference_media_id,
                "reference_observation_id": item.reference_observation_id,
                "source": item.source,
                "source_dataset_key": item.source_dataset_key,
                "source_media_sha256": item.source_media_sha256,
                "reference_bank_version": item.reference_bank_version,
                "admission_policy_fingerprint": item.admission_policy_fingerprint,
                "embedding_fingerprint": item.embedding_fingerprint,
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
                "reference_quality_flags": list(item.reference_quality_flags),
                "matched_escalation_group_ids": group_ids,
                "matched_escalation_decision_fingerprints": decision_fingerprints,
                "trigger_reasons": reasons,
                "recommended_actions": actions,
                "reference_identity_conclusion": "not_assessed",
                "review_status": "pending",
                "human_review_required": True,
                "automatic_reference_exclusion": False,
                "current_support_disposition": item.current_support_disposition,
                "support_disposition_after_targeting": item.current_support_disposition,
                "authorizes_occurrence_release": False,
            }
        )
    semantic_rows.sort(
        key=lambda row: (-float(row["priority_score"]), str(row["reference_media_id"]))
    )
    for rank, row in enumerate(semantic_rows, start=1):
        row["priority_rank"] = rank
    row_fingerprints = [canonical_semantic_fingerprint(row) for row in semantic_rows]
    flagged_ids = set(flagged["group_id"].to_list())
    unmatched_group_ids = tuple(sorted(flagged_ids - matched_group_ids))
    queue_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_VERSION,
            "escalation_report_fingerprint": escalations["report_fingerprint"].item(0),
            "source_candidate_fingerprints": [item.fingerprint for item in items],
            "queue_row_fingerprints": row_fingerprints,
            "matched_escalation_group_ids": sorted(matched_group_ids),
            "unmatched_escalation_group_ids": unmatched_group_ids,
        }
    )
    rows = [
        {
            "schema_version": DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_VERSION,
            "queue_fingerprint": queue_fingerprint,
            "queue_row_fingerprint": row_fingerprint,
            **row,
        }
        for row, row_fingerprint in zip(semantic_rows, row_fingerprints, strict=True)
    ]
    queue = (
        pl.DataFrame(
            rows,
            schema=DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_SCHEMA,
            strict=True,
        )
        if rows
        else pl.DataFrame(schema=DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_SCHEMA)
    )
    validate_dynamic_pool_targeted_reference_review_queue(queue)
    projection_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_TARGETED_REFERENCE_PROJECTION_VERSION,
            "queue_fingerprint": queue_fingerprint,
            "source_candidate_count": len(items),
            "targeted_candidate_count": queue.height,
            "matched_escalation_group_ids": sorted(matched_group_ids),
            "unmatched_escalation_group_ids": unmatched_group_ids,
        }
    )
    return DynamicPoolTargetedReferenceProjection(
        queue=queue,
        projection_fingerprint=projection_fingerprint,
        source_candidate_count=len(items),
        targeted_candidate_count=queue.height,
        matched_escalation_group_ids=tuple(sorted(matched_group_ids)),
        unmatched_escalation_group_ids=unmatched_group_ids,
    )


def validate_dynamic_pool_targeted_reference_review_queue(
    queue: pl.DataFrame,
) -> None:
    """Require pending human review and prohibit automatic reference claims."""

    if not isinstance(queue, pl.DataFrame):
        raise TypeError("targeted reference review queue must be a Polars DataFrame")
    if queue.schema != DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_SCHEMA:
        raise ValueError(
            "targeted reference review queue schema does not match contract"
        )
    if queue.is_empty():
        return
    if queue["reference_media_id"].n_unique() != queue.height:
        raise ValueError("targeted reference review queue repeats a reference media ID")
    if queue["priority_rank"].to_list() != list(range(1, queue.height + 1)):
        raise ValueError("targeted reference review ranks must be contiguous")
    if not queue.equals(
        queue.sort(
            ["priority_score", "reference_media_id"],
            descending=[True, False],
        )
    ):
        raise ValueError("targeted reference review queue is not canonically sorted")
    if queue.filter(
        (pl.col("schema_version") != DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_VERSION)
        | (pl.col("source") != "gbif")
        | (
            pl.col("priority_score_semantics")
            != "transparent_review_heuristic_not_probability"
        )
        | (pl.col("reference_identity_conclusion") != "not_assessed")
        | (pl.col("review_status") != "pending")
        | ~pl.col("human_review_required")
        | pl.col("automatic_reference_exclusion")
        | (
            pl.col("current_support_disposition")
            != pl.col("support_disposition_after_targeting")
        )
        | pl.col("authorizes_occurrence_release")
        | (pl.col("matched_escalation_group_ids").list.len() == 0)
    ).height:
        raise ValueError("targeted reference review crossed its authority contract")
    for row in queue.iter_rows(named=True):
        score = float(row["priority_score"])
        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("targeted reference priority score is invalid")
        components = row["priority_components"]
        expected_components = {
            "embedding_outlier_score",
            "prototype_influence",
            "repeated_error_involvement",
            "route_domain_mismatch",
            "local_scope_member",
            "reference_quality_flag_presence",
        }
        if {
            str(component["component"]) for component in components
        } != expected_components:
            raise ValueError("targeted reference priority components are incomplete")
        base = {
            field: row[field]
            for field in DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_SCHEMA
            if field
            not in {"schema_version", "queue_fingerprint", "queue_row_fingerprint"}
        }
        if row["queue_row_fingerprint"] != canonical_semantic_fingerprint(base):
            raise ValueError("targeted reference queue row fingerprint mismatch")
    if queue["queue_fingerprint"].n_unique() != 1:
        raise ValueError("targeted reference review queue has mixed fingerprints")


def _candidate_matches_escalation(
    candidate: DynamicPoolReferenceReviewCandidate,
    escalation: dict[str, object],
) -> bool:
    hierarchy = str(escalation["hierarchy_level"])
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
    candidate: DynamicPoolReferenceReviewCandidate,
) -> list[dict[str, object]]:
    values = (
        ("embedding_outlier_score", candidate.embedding_outlier_score),
        ("prototype_influence", candidate.prototype_influence),
        (
            "repeated_error_involvement",
            min(candidate.repeated_error_involvement_count / 3.0, 1.0),
        ),
        ("route_domain_mismatch", float(candidate.route_domain_mismatch)),
        ("local_scope_member", float(candidate.local_scope_member)),
        (
            "reference_quality_flag_presence",
            float(bool(candidate.reference_quality_flags)),
        ),
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
    candidates: Sequence[DynamicPoolReferenceReviewCandidate],
) -> tuple[DynamicPoolReferenceReviewCandidate, ...]:
    items = tuple(candidates)
    if any(not isinstance(item, DynamicPoolReferenceReviewCandidate) for item in items):
        raise TypeError("reference candidates must use the dynamic-pool contract")
    ordered = tuple(sorted(items, key=lambda item: item.reference_media_id))
    if len({item.reference_media_id for item in ordered}) != len(ordered):
        raise ValueError("reference candidate media IDs must be unique")
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


def _probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return result


__all__ = [
    "DYNAMIC_POOL_REFERENCE_REVIEW_CANDIDATE_VERSION",
    "DYNAMIC_POOL_TARGETED_REFERENCE_PROJECTION_VERSION",
    "DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_SCHEMA",
    "DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_VERSION",
    "REFERENCE_PRIORITY_COMPONENT_SCHEMA",
    "DynamicPoolReferenceReviewCandidate",
    "DynamicPoolTargetedReferenceProjection",
    "build_dynamic_pool_targeted_reference_review_queue",
    "validate_dynamic_pool_targeted_reference_review_queue",
]
