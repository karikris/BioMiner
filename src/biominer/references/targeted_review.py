"""Rank references for review only after a species-level escalation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path

import polars as pl

from biominer.bioclip.reference_quality_diagnostics import (
    validate_reference_quality_diagnostics,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.reference_escalation import validate_reference_escalations
from biominer.references.schemas import (
    reference_review_queue_schema,
    validate_reference_review_queue,
)
from biominer.storage.parquet import write_parquet


TARGETED_REFERENCE_REVIEW_QUEUE_FILE = "targeted_reference_review_queue.parquet"
TARGETED_REFERENCE_REVIEW_SCHEMA_VERSION = (
    "targeted-reference-review-queue-v1.0.0"
)

FLAGGED_CONTEXT_SCHEMA = pl.Struct(
    {
        "competitor_species": pl.String,
        "region": pl.String,
        "route": pl.String,
        "flag_reasons": pl.List(pl.String),
        "policy_version": pl.String,
        "policy_fingerprint": pl.String,
        "decision_fingerprint": pl.String,
    }
)
PRIORITY_COMPONENT_SCHEMA = pl.Struct(
    {
        "component": pl.String,
        "observed": pl.Float64,
        "normalized_score": pl.Float64,
        "weight": pl.Float64,
    }
)

REFERENCE_ERROR_INVOLVEMENT_SCHEMA = {
    "reference_media_id": pl.String,
    "error_involvement_count": pl.UInt32,
}


def targeted_reference_review_queue_schema() -> dict[str, pl.DataType]:
    return {
        **reference_review_queue_schema(),
        "targeting_schema_version": pl.String,
        "targeting_policy_version": pl.String,
        "targeting_policy_fingerprint": pl.String,
        "flag_reasons": pl.List(pl.String),
        "flagged_contexts": pl.List(FLAGGED_CONTEXT_SCHEMA),
        "embedding_outlier_score": pl.Float64,
        "nearest_competing_species_similarity": pl.Float64,
        "prototype_influence": pl.Float64,
        "route_domain_mismatch": pl.Boolean,
        "provider_dataset_key": pl.String,
        "provider_dataset_concentration": pl.Float64,
        "error_involvement_count": pl.UInt32,
        "subject_area_ratio": pl.Float64,
        "duplicate_ambiguity": pl.Boolean,
        "priority_components": pl.List(PRIORITY_COMPONENT_SCHEMA),
        "target_review_priority_score": pl.Float64,
        "target_review_rank": pl.UInt32,
        "taxon_misidentification_conclusion": pl.String,
        "targeting_fingerprint": pl.String,
    }


@dataclass(frozen=True, slots=True)
class TargetedReferenceReviewPolicy:
    schema_version: str = "targeted-reference-review-policy-v1.0.0"
    policy_version: str = "targeted-reference-priority-v1"
    outlier_weight: float = 1.0
    competitor_similarity_weight: float = 1.0
    prototype_influence_weight: float = 1.0
    route_mismatch_weight: float = 1.0
    provider_concentration_weight: float = 1.0
    repeated_error_weight: float = 1.0
    low_subject_area_weight: float = 1.0
    duplicate_ambiguity_weight: float = 1.0
    repeated_error_saturation_count: int = 3

    def __post_init__(self) -> None:
        for field in fields(self):
            if not field.name.endswith("_weight"):
                continue
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field.name} must be finite and positive")
            object.__setattr__(self, field.name, float(value))
        if (
            isinstance(self.repeated_error_saturation_count, bool)
            or not isinstance(self.repeated_error_saturation_count, int)
            or self.repeated_error_saturation_count < 1
        ):
            raise ValueError("repeated_error_saturation_count must be positive")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {field.name: getattr(self, field.name) for field in fields(self)}
        )


def reference_error_involvement_frame(
    rows: list[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    frame = pl.DataFrame(
        rows or [],
        schema=REFERENCE_ERROR_INVOLVEMENT_SCHEMA,
        orient="row",
        strict=True,
    ).sort("reference_media_id")
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("reference error involvement repeats a reference media ID")
    if frame.filter(pl.col("reference_media_id").str.strip_chars() == "").height:
        raise ValueError("reference error involvement contains a blank media ID")
    return frame


def build_targeted_reference_review_queue(
    base_queue: pl.DataFrame,
    escalations: pl.DataFrame,
    diagnostics: pl.DataFrame,
    support_manifest: pl.DataFrame,
    reference_qa: pl.DataFrame,
    identity_groups: pl.DataFrame,
    error_involvement: pl.DataFrame,
    *,
    policy: TargetedReferenceReviewPolicy | None = None,
) -> pl.DataFrame:
    """Create one transparent priority row per reference in flagged groups."""

    active = policy or TargetedReferenceReviewPolicy()
    validate_reference_review_queue(base_queue)
    _validate_escalations(escalations)
    validate_reference_quality_diagnostics(diagnostics)
    _require_unique_columns(
        support_manifest,
        {
            "reference_media_id",
            "scientific_name",
            "route",
            "support_eligible",
            "provider_dataset_key",
        },
        artifact="reference support manifest",
    )
    _require_unique_columns(
        reference_qa,
        {"reference_media_id", "subject_area_ratio"},
        artifact="reference QA",
    )
    _require_unique_columns(
        identity_groups,
        {"reference_media_id", "resolution_status", "support_disposition"},
        artifact="reference identity groups",
    )
    if error_involvement.schema != REFERENCE_ERROR_INVOLVEMENT_SCHEMA:
        raise ValueError("reference error involvement schema mismatch")
    reference_error_involvement_frame(error_involvement.to_dicts())

    flagged_contexts = _flagged_contexts(escalations)
    if not flagged_contexts:
        return pl.DataFrame(schema=targeted_reference_review_queue_schema())

    queue_by_id = _rows_by_id(base_queue, artifact="base review queue")
    support_by_id = _rows_by_id(
        support_manifest,
        artifact="reference support manifest",
    )
    qa_by_id = _rows_by_id(reference_qa, artifact="reference QA")
    identity_by_id = _rows_by_id(
        identity_groups,
        artifact="reference identity groups",
    )
    errors_by_id = {
        str(row["reference_media_id"]): int(row["error_involvement_count"])
        for row in error_involvement.iter_rows(named=True)
    }
    provider_concentrations = _provider_concentrations(support_manifest)

    ranked: list[dict[str, object]] = []
    for diagnostic in diagnostics.iter_rows(named=True):
        group = (str(diagnostic["species"]), str(diagnostic["route"]))
        contexts = flagged_contexts.get(group)
        if contexts is None:
            continue
        media_id = str(diagnostic["reference_media_id"])
        queue_row = _required_row(queue_by_id, media_id, artifact="base review queue")
        support = _required_row(
            support_by_id,
            media_id,
            artifact="reference support manifest",
        )
        qa = _required_row(qa_by_id, media_id, artifact="reference QA")
        identity = _required_row(
            identity_by_id,
            media_id,
            artifact="reference identity groups",
        )
        _validate_row_bindings(
            queue_row,
            diagnostic,
            support,
            expected_group=group,
        )
        provider_key = _required_text(
            support["provider_dataset_key"],
            field="provider_dataset_key",
        )
        concentration_key = (group[0], group[1], provider_key)
        if concentration_key not in provider_concentrations:
            raise ValueError(
                "provider concentration unavailable for targeted reference "
                + media_id
            )
        subject_area = _optional_unit_interval(
            qa["subject_area_ratio"],
            field="subject_area_ratio",
        )
        duplicate_ambiguity = (
            str(identity["resolution_status"]) != "resolved"
            or str(identity["support_disposition"])
            in {"unresolved_duplicate", "duplicate_conflict"}
        )
        error_count = errors_by_id.get(media_id, 0)
        components = _priority_components(
            diagnostic=diagnostic,
            provider_concentration=provider_concentrations[concentration_key],
            error_count=error_count,
            subject_area_ratio=subject_area,
            duplicate_ambiguity=duplicate_ambiguity,
            policy=active,
        )
        score = sum(
            float(component["normalized_score"]) * float(component["weight"])
            for component in components
        ) / sum(float(component["weight"]) for component in components)
        reasons = sorted(
            {
                str(reason)
                for context in contexts
                for reason in context["flag_reasons"]  # type: ignore[union-attr]
            }
        )
        ranked.append(
            {
                **queue_row,
                "targeting_schema_version": (
                    TARGETED_REFERENCE_REVIEW_SCHEMA_VERSION
                ),
                "targeting_policy_version": active.policy_version,
                "targeting_policy_fingerprint": active.fingerprint,
                "flag_reasons": reasons,
                "flagged_contexts": contexts,
                "embedding_outlier_score": diagnostic[
                    "embedding_outlier_score"
                ],
                "nearest_competing_species_similarity": diagnostic[
                    "nearest_competing_species_similarity"
                ],
                "prototype_influence": diagnostic["prototype_influence"],
                "route_domain_mismatch": diagnostic["route_domain_mismatch"],
                "provider_dataset_key": provider_key,
                "provider_dataset_concentration": provider_concentrations[
                    concentration_key
                ],
                "error_involvement_count": error_count,
                "subject_area_ratio": subject_area,
                "duplicate_ambiguity": duplicate_ambiguity,
                "priority_components": components,
                "target_review_priority_score": score,
                "target_review_rank": 0,
                "taxon_misidentification_conclusion": "not_assessed",
                "targeting_fingerprint": "",
            }
        )

    ranked.sort(
        key=lambda row: (
            -float(row["target_review_priority_score"]),
            str(row["reference_media_id"]),
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["target_review_rank"] = rank
        row["review_priority"] = rank
        row["review_reason"] = (
            str(row["review_reason"])
            + ";statistically_flagged_reference:"
            + ",".join(row["flag_reasons"])  # type: ignore[arg-type]
        )
        payload = dict(row)
        payload.pop("targeting_fingerprint")
        row["targeting_fingerprint"] = canonical_semantic_fingerprint(payload)

    result = pl.DataFrame(
        ranked,
        schema=targeted_reference_review_queue_schema(),
        orient="row",
        strict=True,
    ).sort("target_review_rank", "reference_media_id")
    validate_targeted_reference_review_queue(result)
    return result


def validate_targeted_reference_review_queue(frame: pl.DataFrame) -> None:
    if frame.schema != targeted_reference_review_queue_schema():
        raise ValueError("targeted reference review queue schema mismatch")
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("targeted reference review queue repeats a media ID")
    if frame["review_request_id"].n_unique() != frame.height:
        raise ValueError("targeted reference review queue repeats a request ID")
    expected_order = frame.sort("target_review_rank", "reference_media_id")
    if not frame.equals(expected_order):
        raise ValueError("targeted reference review queue is not deterministically sorted")
    if frame["target_review_rank"].to_list() != list(range(1, frame.height + 1)):
        raise ValueError("targeted reference review ranks must be contiguous")

    base = frame.select(list(reference_review_queue_schema()))
    validate_reference_review_queue(base)
    expected_components = {
        "embedding_outlier_score",
        "nearest_competing_species_similarity",
        "prototype_influence",
        "route_domain_mismatch",
        "provider_dataset_concentration",
        "repeated_error_involvement",
        "low_subject_area_ratio",
        "duplicate_ambiguity",
    }
    for row in frame.iter_rows(named=True):
        if (
            row["targeting_schema_version"]
            != TARGETED_REFERENCE_REVIEW_SCHEMA_VERSION
            or row["taxon_misidentification_conclusion"] != "not_assessed"
            or not row["flag_reasons"]
            or not row["flagged_contexts"]
        ):
            raise ValueError("targeted reference review semantics are invalid")
        score = float(row["target_review_priority_score"])
        if not isfinite(score) or not 0 <= score <= 1:
            raise ValueError("targeted reference priority score is invalid")
        components = row["priority_components"]
        if {item["component"] for item in components} != expected_components:
            raise ValueError("targeted reference priority components are incomplete")
        for item in components:
            component_score = float(item["normalized_score"])
            weight = float(item["weight"])
            if (
                not isfinite(component_score)
                or not 0 <= component_score <= 1
                or not isfinite(weight)
                or weight <= 0
            ):
                raise ValueError("targeted reference priority component is invalid")
        payload = dict(row)
        fingerprint = payload.pop("targeting_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("targeted reference queue fingerprint mismatch")


def write_targeted_reference_review_queue(
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    validate_targeted_reference_review_queue(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= TARGETED_REFERENCE_REVIEW_QUEUE_FILE
    return write_parquet(frame, destination)


def _validate_escalations(escalations: pl.DataFrame) -> None:
    validate_reference_escalations(escalations)


def _flagged_contexts(
    escalations: pl.DataFrame,
) -> dict[tuple[str, str], list[dict[str, object]]]:
    result: dict[tuple[str, str], list[dict[str, object]]] = {}
    flagged = escalations.filter(pl.col("flagged_for_reference_review"))
    for row in flagged.sort(
        "target_species", "route", "competitor_species", "region"
    ).iter_rows(named=True):
        key = (str(row["target_species"]), str(row["route"]))
        result.setdefault(key, []).append(
            {
                "competitor_species": row["competitor_species"],
                "region": row["region"],
                "route": row["route"],
                "flag_reasons": sorted(set(row["flag_reasons"])),
                "policy_version": row["policy_version"],
                "policy_fingerprint": row["policy_fingerprint"],
                "decision_fingerprint": row["decision_fingerprint"],
            }
        )
    return result


def _provider_concentrations(
    support_manifest: pl.DataFrame,
) -> dict[tuple[str, str, str], float]:
    eligible = support_manifest.filter(pl.col("support_eligible"))
    totals: Counter[tuple[str, str]] = Counter()
    providers: Counter[tuple[str, str, str]] = Counter()
    for row in eligible.iter_rows(named=True):
        group = (str(row["scientific_name"]), str(row["route"]))
        provider = _required_text(
            row["provider_dataset_key"],
            field="provider_dataset_key",
        )
        totals[group] += 1
        providers[(*group, provider)] += 1
    return {
        key: count / totals[(key[0], key[1])]
        for key, count in providers.items()
    }


def _priority_components(
    *,
    diagnostic: Mapping[str, object],
    provider_concentration: float,
    error_count: int,
    subject_area_ratio: float | None,
    duplicate_ambiguity: bool,
    policy: TargetedReferenceReviewPolicy,
) -> list[dict[str, object]]:
    outlier = _bounded(
        diagnostic["embedding_outlier_score"],
        lower=0,
        upper=2,
        field="embedding_outlier_score",
    )
    competitor = _optional_bounded(
        diagnostic["nearest_competing_species_similarity"],
        lower=-1,
        upper=1,
        field="nearest_competing_species_similarity",
    )
    influence = _optional_bounded(
        diagnostic["prototype_influence"],
        lower=0,
        upper=2,
        field="prototype_influence",
    )
    values = (
        (
            "embedding_outlier_score",
            outlier,
            outlier / 2,
            policy.outlier_weight,
        ),
        (
            "nearest_competing_species_similarity",
            competitor,
            (competitor + 1) / 2 if competitor is not None else 0.0,
            policy.competitor_similarity_weight,
        ),
        (
            "prototype_influence",
            influence,
            influence / 2 if influence is not None else 0.0,
            policy.prototype_influence_weight,
        ),
        (
            "route_domain_mismatch",
            float(bool(diagnostic["route_domain_mismatch"])),
            float(bool(diagnostic["route_domain_mismatch"])),
            policy.route_mismatch_weight,
        ),
        (
            "provider_dataset_concentration",
            provider_concentration,
            provider_concentration,
            policy.provider_concentration_weight,
        ),
        (
            "repeated_error_involvement",
            float(error_count),
            min(error_count / policy.repeated_error_saturation_count, 1.0),
            policy.repeated_error_weight,
        ),
        (
            "low_subject_area_ratio",
            subject_area_ratio,
            1 - subject_area_ratio if subject_area_ratio is not None else 0.0,
            policy.low_subject_area_weight,
        ),
        (
            "duplicate_ambiguity",
            float(duplicate_ambiguity),
            float(duplicate_ambiguity),
            policy.duplicate_ambiguity_weight,
        ),
    )
    return [
        {
            "component": component,
            "observed": observed,
            "normalized_score": normalized,
            "weight": weight,
        }
        for component, observed, normalized, weight in values
    ]


def _validate_row_bindings(
    queue_row: Mapping[str, object],
    diagnostic: Mapping[str, object],
    support: Mapping[str, object],
    *,
    expected_group: tuple[str, str],
) -> None:
    media_ids = {
        str(queue_row["reference_media_id"]),
        str(diagnostic["reference_media_id"]),
        str(support["reference_media_id"]),
    }
    if len(media_ids) != 1:
        raise ValueError("targeted reference evidence media bindings disagree")
    if (
        queue_row["scientific_name"] != expected_group[0]
        or support["scientific_name"] != expected_group[0]
        or support["route"] != expected_group[1]
    ):
        raise ValueError("targeted reference species or route bindings disagree")
    if not support["support_eligible"]:
        raise ValueError("targeted reference must be support eligible")


def _rows_by_id(
    frame: pl.DataFrame,
    *,
    artifact: str,
) -> dict[str, dict[str, object]]:
    result = {
        str(row["reference_media_id"]): row
        for row in frame.iter_rows(named=True)
    }
    if len(result) != frame.height:
        raise ValueError(f"{artifact} repeats a reference media ID")
    return result


def _required_row(
    rows: Mapping[str, dict[str, object]],
    media_id: str,
    *,
    artifact: str,
) -> dict[str, object]:
    row = rows.get(media_id)
    if row is None:
        raise ValueError(f"{artifact} missing targeted reference {media_id}")
    return row


def _require_unique_columns(
    frame: pl.DataFrame,
    required: set[str],
    *,
    artifact: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{artifact} missing columns: {missing}")
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError(f"{artifact} repeats a reference media ID")


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _optional_unit_interval(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _bounded(value, lower=0, upper=1, field=field)


def _optional_bounded(
    value: object,
    *,
    lower: float,
    upper: float,
    field: str,
) -> float | None:
    if value is None:
        return None
    return _bounded(value, lower=lower, upper=upper, field=field)


def _bounded(
    value: object,
    *,
    lower: float,
    upper: float,
    field: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{field} must be finite within [{lower}, {upper}]")
    return number


__all__ = [
    "REFERENCE_ERROR_INVOLVEMENT_SCHEMA",
    "TARGETED_REFERENCE_REVIEW_QUEUE_FILE",
    "TARGETED_REFERENCE_REVIEW_SCHEMA_VERSION",
    "TargetedReferenceReviewPolicy",
    "build_targeted_reference_review_queue",
    "reference_error_involvement_frame",
    "targeted_reference_review_queue_schema",
    "validate_targeted_reference_review_queue",
    "write_targeted_reference_review_queue",
]
