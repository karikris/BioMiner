"""Selective Flickr rescoring after an adaptive reference-bank revision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.target_aware_output import (
    validate_target_aware_object_scores,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.adaptive_bank_revision import (
    AdaptiveSupportBankRevision,
    validate_adaptive_support_bank_revision,
)
from biominer.storage.parquet import write_parquet


FLICKR_RESCORE_PLAN_FILE = "flickr_rescore_plan.parquet"
SCORE_REFERENCE_DEPENDENCY_SCHEMA_VERSION = (
    "score-reference-dependency-v1.0.0"
)
FLICKR_RESCORE_EVIDENCE_SCHEMA_VERSION = "flickr-rescore-evidence-v1.0.0"
FLICKR_RESCORE_PLAN_SCHEMA_VERSION = "flickr-rescore-plan-v1.0.0"
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


SCORE_REFERENCE_DEPENDENCY_SCHEMA = {
    "schema_version": pl.String,
    "target_score_id": pl.String,
    "reference_media_ids": pl.List(pl.String),
    "dependency_fingerprint": pl.String,
}

FLICKR_RESCORE_EVIDENCE_SCHEMA = {
    "schema_version": pl.String,
    "target_score_id": pl.String,
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "scoring_unit_id": pl.String,
    "route": pl.String,
    "prior_target_score_fingerprint": pl.String,
    "prior_reference_bank_fingerprint": pl.String,
    "target_accepted_taxon_key": pl.String,
    "best_competitor_accepted_taxon_key": pl.String,
    "candidate_accepted_taxon_keys": pl.List(pl.String),
    "reference_media_ids": pl.List(pl.String),
    "reference_dependencies_complete": pl.Boolean,
    "prior_target_competitor_margin": pl.Float64,
    "evidence_fingerprint": pl.String,
}


def flickr_rescore_plan_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "revision_fingerprint": pl.String,
        **{
            field: dtype
            for field, dtype in FLICKR_RESCORE_EVIDENCE_SCHEMA.items()
            if field not in {"schema_version", "evidence_fingerprint"}
        },
        "evidence_fingerprint": pl.String,
        "margin_impact_band": pl.Float64,
        "target_bank_changed": pl.Boolean,
        "best_competitor_bank_changed": pl.Boolean,
        "candidate_union_changed": pl.Boolean,
        "removed_reference_dependency": pl.Boolean,
        "margin_in_impact_band": pl.Boolean,
        "rescore_required": pl.Boolean,
        "rescore_reasons": pl.List(pl.String),
        "rescore_action": pl.String,
        "plan_fingerprint": pl.String,
    }


def score_reference_dependency_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for source in rows or ():
        row = dict(source)
        row.setdefault(
            "schema_version",
            SCORE_REFERENCE_DEPENDENCY_SCHEMA_VERSION,
        )
        row["reference_media_ids"] = sorted(set(row["reference_media_ids"]))
        row["dependency_fingerprint"] = ""
        payload = dict(row)
        payload.pop("dependency_fingerprint")
        row["dependency_fingerprint"] = canonical_semantic_fingerprint(payload)
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=SCORE_REFERENCE_DEPENDENCY_SCHEMA,
        orient="row",
        strict=True,
    ).sort("target_score_id")
    validate_score_reference_dependencies(frame)
    return frame


def flickr_rescore_evidence_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for source in rows or ():
        row = dict(source)
        row.setdefault("schema_version", FLICKR_RESCORE_EVIDENCE_SCHEMA_VERSION)
        for field in ("candidate_accepted_taxon_keys", "reference_media_ids"):
            row[field] = sorted(set(row[field]))
        row["evidence_fingerprint"] = ""
        payload = dict(row)
        payload.pop("evidence_fingerprint")
        row["evidence_fingerprint"] = canonical_semantic_fingerprint(payload)
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=FLICKR_RESCORE_EVIDENCE_SCHEMA,
        orient="row",
        strict=True,
    ).sort("target_score_id")
    validate_flickr_rescore_evidence(frame)
    return frame


def flickr_rescore_evidence_from_target_scores(
    prior_scores: pl.DataFrame,
    reference_dependencies: pl.DataFrame,
) -> pl.DataFrame:
    """Project production score provenance into the incremental selector."""

    validate_target_aware_object_scores(prior_scores)
    validate_score_reference_dependencies(reference_dependencies)
    score_ids = set(prior_scores["target_score_id"])
    unknown = sorted(set(reference_dependencies["target_score_id"]) - score_ids)
    if unknown:
        raise ValueError(
            "score reference dependencies name unknown target scores: "
            + ", ".join(unknown)
        )
    dependencies_by_id = {
        str(row["target_score_id"]): row
        for row in reference_dependencies.iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    for score in prior_scores.iter_rows(named=True):
        score_id = str(score["target_score_id"])
        dependency = dependencies_by_id.get(score_id)
        rows.append(
            {
                "target_score_id": score_id,
                "source": score["source"],
                "flickr_photo_id": score["flickr_photo_id"],
                "scoring_unit_id": score["scoring_unit_id"],
                "route": score["route"],
                "prior_target_score_fingerprint": score[
                    "target_score_fingerprint"
                ],
                "prior_reference_bank_fingerprint": score[
                    "reference_bank_fingerprint"
                ],
                "target_accepted_taxon_key": score[
                    "target_accepted_taxon_key"
                ],
                "best_competitor_accepted_taxon_key": score[
                    "best_competitor_accepted_taxon_key"
                ],
                "candidate_accepted_taxon_keys": [
                    candidate["accepted_taxon_key"]
                    for candidate in score["regional_candidate_evidence"]
                ],
                "reference_media_ids": (
                    dependency["reference_media_ids"] if dependency else []
                ),
                "reference_dependencies_complete": dependency is not None,
                "prior_target_competitor_margin": score[
                    "target_competitor_margin"
                ],
            }
        )
    return flickr_rescore_evidence_frame(rows)


def calculate_flickr_rescore_plan(
    revision: AdaptiveSupportBankRevision,
    evidence: pl.DataFrame,
    *,
    margin_impact_band: float,
) -> pl.DataFrame:
    """Select records touched by reference changes or uncertain prior margins."""

    validate_adaptive_support_bank_revision(revision)
    validate_flickr_rescore_evidence(evidence)
    band = _margin_band(margin_impact_band)
    if evidence.filter(
        pl.col("prior_reference_bank_fingerprint")
        != revision.old_reference_bank_fingerprint
    ).height:
        raise ValueError("Flickr rescore evidence is bound to the wrong reference bank")
    changed_species = {
        str(row["accepted_taxon_key"])
        for row in revision.change_manifest.iter_rows(named=True)
        if not str(row["change_type"]).startswith("unchanged")
    }
    removed_references = {
        str(row["reference_media_id"])
        for row in revision.change_manifest.iter_rows(named=True)
        if row["old_support_eligible"] and not row["new_support_eligible"]
    }
    rows: list[dict[str, object]] = []
    for source in evidence.iter_rows(named=True):
        candidate_keys = set(source["candidate_accepted_taxon_keys"])
        reference_ids = set(source["reference_media_ids"])
        margin = source["prior_target_competitor_margin"]
        flags = {
            "target_bank_changed": (
                source["target_accepted_taxon_key"] in changed_species
            ),
            "best_competitor_bank_changed": (
                source["best_competitor_accepted_taxon_key"] in changed_species
            ),
            "candidate_union_changed": bool(candidate_keys & changed_species),
            "removed_reference_dependency": bool(
                reference_ids & removed_references
            ),
            "margin_in_impact_band": (
                margin is not None and abs(float(margin)) <= band
            ),
        }
        reasons = sorted(
            reason for reason, triggered in flags.items() if triggered
        )
        if not source["reference_dependencies_complete"]:
            reasons.append("missing_reference_dependency_evidence")
        if margin is None:
            reasons.append("missing_prior_margin_evidence")
        reasons.sort()
        row = {
            "schema_version": FLICKR_RESCORE_PLAN_SCHEMA_VERSION,
            "revision_fingerprint": revision.revision_fingerprint,
            **{
                field: source[field]
                for field in FLICKR_RESCORE_EVIDENCE_SCHEMA
                if field != "schema_version"
            },
            "margin_impact_band": band,
            **flags,
            "rescore_required": bool(reasons),
            "rescore_reasons": reasons,
            "rescore_action": (
                "selectively_rescore" if reasons else "reuse_prior_score"
            ),
            "plan_fingerprint": "",
        }
        payload = dict(row)
        payload.pop("plan_fingerprint")
        row["plan_fingerprint"] = canonical_semantic_fingerprint(payload)
        rows.append(row)
    plan = pl.DataFrame(
        rows,
        schema=flickr_rescore_plan_schema(),
        orient="row",
        strict=True,
    ).sort("target_score_id")
    validate_flickr_rescore_plan(plan)
    return plan


def target_score_ids_to_rescore(plan: pl.DataFrame) -> tuple[str, ...]:
    validate_flickr_rescore_plan(plan)
    return tuple(
        plan.filter(pl.col("rescore_required"))["target_score_id"].to_list()
    )


def flickr_photo_ids_to_rescore(plan: pl.DataFrame) -> tuple[str, ...]:
    validate_flickr_rescore_plan(plan)
    return tuple(
        sorted(
            set(
                plan.filter(pl.col("rescore_required"))[
                    "flickr_photo_id"
                ].to_list()
            )
        )
    )


def flickr_rescore_metrics(plan: pl.DataFrame) -> pl.DataFrame:
    validate_flickr_rescore_plan(plan)
    return (
        plan.group_by("rescore_action")
        .agg(pl.len().cast(pl.UInt64).alias("record_count"))
        .sort("rescore_action")
    )


def validate_score_reference_dependencies(frame: pl.DataFrame) -> None:
    if frame.schema != SCORE_REFERENCE_DEPENDENCY_SCHEMA:
        raise ValueError("score reference dependency schema mismatch")
    _reject_nulls(frame, tuple(SCORE_REFERENCE_DEPENDENCY_SCHEMA))
    if not frame.equals(frame.sort("target_score_id")):
        raise ValueError("score reference dependencies are not sorted")
    if frame["target_score_id"].n_unique() != frame.height:
        raise ValueError("score reference dependencies repeat a target score")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != SCORE_REFERENCE_DEPENDENCY_SCHEMA_VERSION:
            raise ValueError("unsupported score reference dependency version")
        _required_text(row["target_score_id"], field="target_score_id")
        _canonical_text_list(row["reference_media_ids"], field="reference_media_ids")
        _validate_fingerprint(row, field="dependency_fingerprint")


def validate_flickr_rescore_evidence(frame: pl.DataFrame) -> None:
    if frame.schema != FLICKR_RESCORE_EVIDENCE_SCHEMA:
        raise ValueError("Flickr rescore evidence schema mismatch")
    required = tuple(
        field
        for field in FLICKR_RESCORE_EVIDENCE_SCHEMA
        if field != "prior_target_competitor_margin"
    )
    _reject_nulls(frame, required)
    if not frame.equals(frame.sort("target_score_id")):
        raise ValueError("Flickr rescore evidence is not sorted")
    if frame["target_score_id"].n_unique() != frame.height:
        raise ValueError("Flickr rescore evidence repeats a target score")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != FLICKR_RESCORE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported Flickr rescore evidence version")
        for field in (
            "target_score_id",
            "source",
            "flickr_photo_id",
            "scoring_unit_id",
            "route",
            "target_accepted_taxon_key",
            "best_competitor_accepted_taxon_key",
        ):
            _required_text(row[field], field=field)
        for field in (
            "prior_target_score_fingerprint",
            "prior_reference_bank_fingerprint",
        ):
            _sha256(row[field], field=field)
        candidates = _canonical_text_list(
            row["candidate_accepted_taxon_keys"],
            field="candidate_accepted_taxon_keys",
        )
        if row["target_accepted_taxon_key"] not in candidates:
            raise ValueError("Flickr rescore candidate union omits the target")
        if row["best_competitor_accepted_taxon_key"] not in candidates:
            raise ValueError("Flickr rescore candidate union omits the competitor")
        reference_ids = _canonical_text_list(
            row["reference_media_ids"],
            field="reference_media_ids",
        )
        if not row["reference_dependencies_complete"] and reference_ids:
            raise ValueError(
                "incomplete reference dependencies cannot claim reference IDs"
            )
        margin = row["prior_target_competitor_margin"]
        if margin is not None and not isfinite(float(margin)):
            raise ValueError("prior target competitor margin must be finite")
        _validate_fingerprint(row, field="evidence_fingerprint")


def validate_flickr_rescore_plan(frame: pl.DataFrame) -> None:
    if frame.schema != flickr_rescore_plan_schema():
        raise ValueError("Flickr rescore plan schema mismatch")
    required = tuple(
        field
        for field in flickr_rescore_plan_schema()
        if field != "prior_target_competitor_margin"
    )
    _reject_nulls(frame, required)
    if not frame.equals(frame.sort("target_score_id")):
        raise ValueError("Flickr rescore plan is not sorted")
    if frame["target_score_id"].n_unique() != frame.height:
        raise ValueError("Flickr rescore plan repeats a target score")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != FLICKR_RESCORE_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported Flickr rescore plan version")
        evidence = {
            "schema_version": FLICKR_RESCORE_EVIDENCE_SCHEMA_VERSION,
            **{
                field: row[field]
                for field in FLICKR_RESCORE_EVIDENCE_SCHEMA
                if field != "schema_version"
            },
        }
        validate_flickr_rescore_evidence(
            pl.DataFrame(
                [evidence],
                schema=FLICKR_RESCORE_EVIDENCE_SCHEMA,
                orient="row",
                strict=True,
            )
        )
        band = _margin_band(row["margin_impact_band"])
        margin = row["prior_target_competitor_margin"]
        if row["margin_in_impact_band"] != (
            margin is not None and abs(float(margin)) <= band
        ):
            raise ValueError("Flickr rescore margin-band decision mismatch")
        expected_reasons = sorted(
            field
            for field in (
                "target_bank_changed",
                "best_competitor_bank_changed",
                "candidate_union_changed",
                "removed_reference_dependency",
                "margin_in_impact_band",
            )
            if row[field]
        )
        if not row["reference_dependencies_complete"]:
            expected_reasons.append("missing_reference_dependency_evidence")
        if margin is None:
            expected_reasons.append("missing_prior_margin_evidence")
        expected_reasons.sort()
        if row["rescore_reasons"] != expected_reasons:
            raise ValueError("Flickr rescore reasons mismatch")
        required_rescore = bool(expected_reasons)
        if row["rescore_required"] != required_rescore:
            raise ValueError("Flickr rescore requirement mismatch")
        expected_action = (
            "selectively_rescore" if required_rescore else "reuse_prior_score"
        )
        if row["rescore_action"] != expected_action:
            raise ValueError("Flickr rescore action mismatch")
        _sha256(row["revision_fingerprint"], field="revision_fingerprint")
        _validate_fingerprint(row, field="plan_fingerprint")


def write_flickr_rescore_plan(frame: pl.DataFrame, output: str | Path) -> Path:
    validate_flickr_rescore_plan(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= FLICKR_RESCORE_PLAN_FILE
    return write_parquet(frame, destination)


def _canonical_text_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    normalized = [_required_text(item, field=field) for item in value]
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{field} must be sorted and unique")
    return normalized


def _margin_band(value: object) -> float:
    try:
        band = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("margin impact band must be finite and non-negative") from exc
    if not isfinite(band) or band < 0.0:
        raise ValueError("margin impact band must be finite and non-negative")
    return band


def _validate_fingerprint(row: Mapping[str, object], *, field: str) -> None:
    payload = dict(row)
    fingerprint = payload.pop(field)
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError(f"{field} mismatch")


def _reject_nulls(frame: pl.DataFrame, fields: tuple[str, ...]) -> None:
    if any(frame[field].null_count() for field in fields):
        raise ValueError("Flickr rescore artifacts contain null required fields")


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full sha256 fingerprint")
    return text


__all__ = [
    "FLICKR_RESCORE_EVIDENCE_SCHEMA",
    "FLICKR_RESCORE_EVIDENCE_SCHEMA_VERSION",
    "FLICKR_RESCORE_PLAN_FILE",
    "FLICKR_RESCORE_PLAN_SCHEMA_VERSION",
    "SCORE_REFERENCE_DEPENDENCY_SCHEMA",
    "SCORE_REFERENCE_DEPENDENCY_SCHEMA_VERSION",
    "calculate_flickr_rescore_plan",
    "flickr_photo_ids_to_rescore",
    "flickr_rescore_evidence_frame",
    "flickr_rescore_evidence_from_target_scores",
    "flickr_rescore_metrics",
    "flickr_rescore_plan_schema",
    "score_reference_dependency_frame",
    "target_score_ids_to_rescore",
    "validate_flickr_rescore_evidence",
    "validate_flickr_rescore_plan",
    "validate_score_reference_dependencies",
    "write_flickr_rescore_plan",
]
