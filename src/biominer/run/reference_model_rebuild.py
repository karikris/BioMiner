"""Selective reference-model rebuild planning from revision impact evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.run.reference_revision_impact import (
    validate_reference_revision_impact,
)
from biominer.storage.parquet import write_parquet


REFERENCE_MODEL_REBUILD_PLAN_FILE = "reference_model_rebuild_plan.parquet"
CALIBRATOR_TRAINING_CHANGE_SCHEMA_VERSION = (
    "calibrator-training-change-v1.0.0"
)
REFERENCE_MODEL_REBUILD_PLAN_SCHEMA_VERSION = (
    "reference-model-rebuild-plan-v1.0.0"
)
MODEL_ARTIFACT_TYPES = frozenset(
    {
        "reference_prototype",
        "regional_prototype",
        "candidate_set",
        "classifier",
        "calibrator",
    }
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


CALIBRATOR_TRAINING_CHANGE_SCHEMA = {
    "schema_version": pl.String,
    "artifact_id": pl.String,
    "old_training_data_fingerprint": pl.String,
    "new_training_data_fingerprint": pl.String,
    "training_data_changed": pl.Boolean,
    "change_fingerprint": pl.String,
}


def reference_model_rebuild_plan_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "revision_fingerprint": pl.String,
        "artifact_id": pl.String,
        "artifact_type": pl.String,
        "current_artifact_fingerprint": pl.String,
        "source_impact_fingerprint": pl.String,
        "source_impact_status": pl.String,
        "rebuild_status": pl.String,
        "rebuild_action": pl.String,
        "affected_reference_media_ids": pl.List(pl.String),
        "affected_species": pl.List(pl.String),
        "affected_routes": pl.List(pl.String),
        "affected_regions": pl.List(pl.String),
        "old_training_data_fingerprint": pl.String,
        "new_training_data_fingerprint": pl.String,
        "training_data_changed": pl.Boolean,
        "plan_fingerprint": pl.String,
    }


def calibrator_training_change_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for source in rows or ():
        row = dict(source)
        row.setdefault(
            "schema_version",
            CALIBRATOR_TRAINING_CHANGE_SCHEMA_VERSION,
        )
        row["training_data_changed"] = (
            row.get("old_training_data_fingerprint")
            != row.get("new_training_data_fingerprint")
        )
        row["change_fingerprint"] = ""
        payload = dict(row)
        payload.pop("change_fingerprint")
        row["change_fingerprint"] = canonical_semantic_fingerprint(payload)
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=CALIBRATOR_TRAINING_CHANGE_SCHEMA,
        orient="row",
        strict=True,
    ).sort("artifact_id")
    validate_calibrator_training_changes(frame)
    return frame


def calculate_reference_model_rebuild_plan(
    revision_impact: pl.DataFrame,
    calibrator_training_changes: pl.DataFrame,
) -> pl.DataFrame:
    """Select the exact model artifacts whose semantic inputs changed."""

    validate_reference_revision_impact(revision_impact)
    validate_calibrator_training_changes(calibrator_training_changes)
    model_impact = revision_impact.filter(
        pl.col("artifact_type").is_in(sorted(MODEL_ARTIFACT_TYPES))
    )
    calibrator_ids = set(
        model_impact.filter(pl.col("artifact_type") == "calibrator")[
            "artifact_id"
        ]
    )
    declared_training_ids = set(calibrator_training_changes["artifact_id"])
    unknown_training_ids = sorted(declared_training_ids - calibrator_ids)
    if unknown_training_ids:
        raise ValueError(
            "calibrator training changes reference unknown calibrators: "
            + ", ".join(unknown_training_ids)
        )
    changes_by_id = {
        str(row["artifact_id"]): row
        for row in calibrator_training_changes.iter_rows(named=True)
    }
    affected_calibrators = set(
        model_impact.filter(
            (pl.col("artifact_type") == "calibrator")
            & (pl.col("impact_status") == "affected")
        )["artifact_id"]
    )
    missing_training_ids = sorted(affected_calibrators - declared_training_ids)
    if missing_training_ids:
        raise ValueError(
            "affected calibrators lack training-data change evidence: "
            + ", ".join(missing_training_ids)
        )
    rows: list[dict[str, object]] = []
    for impact in model_impact.iter_rows(named=True):
        artifact_id = str(impact["artifact_id"])
        artifact_type = str(impact["artifact_type"])
        training = changes_by_id.get(artifact_id)
        rebuild, action = _rebuild_decision(impact, training=training)
        row = {
            "schema_version": REFERENCE_MODEL_REBUILD_PLAN_SCHEMA_VERSION,
            "revision_fingerprint": impact["revision_fingerprint"],
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "current_artifact_fingerprint": impact["artifact_fingerprint"],
            "source_impact_fingerprint": impact["impact_fingerprint"],
            "source_impact_status": impact["impact_status"],
            "rebuild_status": "rebuild" if rebuild else "reuse",
            "rebuild_action": action,
            "affected_reference_media_ids": impact[
                "affected_reference_media_ids"
            ],
            "affected_species": impact["affected_species"],
            "affected_routes": impact["affected_routes"],
            "affected_regions": impact["affected_regions"],
            "old_training_data_fingerprint": (
                training["old_training_data_fingerprint"] if training else None
            ),
            "new_training_data_fingerprint": (
                training["new_training_data_fingerprint"] if training else None
            ),
            "training_data_changed": (
                training["training_data_changed"] if training else None
            ),
            "plan_fingerprint": "",
        }
        payload = dict(row)
        payload.pop("plan_fingerprint")
        row["plan_fingerprint"] = canonical_semantic_fingerprint(payload)
        rows.append(row)
    plan = pl.DataFrame(
        rows,
        schema=reference_model_rebuild_plan_schema(),
        orient="row",
        strict=True,
    ).sort("artifact_id")
    validate_reference_model_rebuild_plan(plan)
    return plan


def reference_model_artifact_ids_to_rebuild(
    plan: pl.DataFrame,
    *,
    artifact_type: str | None = None,
) -> tuple[str, ...]:
    validate_reference_model_rebuild_plan(plan)
    selected = plan.filter(pl.col("rebuild_status") == "rebuild")
    if artifact_type is not None:
        kind = _artifact_type(artifact_type)
        selected = selected.filter(pl.col("artifact_type") == kind)
    return tuple(selected["artifact_id"].to_list())


def reference_model_rebuild_metrics(plan: pl.DataFrame) -> pl.DataFrame:
    validate_reference_model_rebuild_plan(plan)
    return (
        plan.group_by("artifact_type", "rebuild_status", "rebuild_action")
        .agg(pl.len().cast(pl.UInt64).alias("artifact_count"))
        .sort("artifact_type", "rebuild_status", "rebuild_action")
    )


def validate_calibrator_training_changes(frame: pl.DataFrame) -> None:
    if frame.schema != CALIBRATOR_TRAINING_CHANGE_SCHEMA:
        raise ValueError("calibrator training change schema mismatch")
    _reject_nulls(frame, tuple(CALIBRATOR_TRAINING_CHANGE_SCHEMA))
    if not frame.equals(frame.sort("artifact_id")):
        raise ValueError("calibrator training changes are not sorted")
    if frame["artifact_id"].n_unique() != frame.height:
        raise ValueError("calibrator training changes repeat an artifact")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != CALIBRATOR_TRAINING_CHANGE_SCHEMA_VERSION:
            raise ValueError("unsupported calibrator training change version")
        _required_text(row["artifact_id"], field="artifact_id")
        old = _sha256(
            row["old_training_data_fingerprint"],
            field="old_training_data_fingerprint",
        )
        new = _sha256(
            row["new_training_data_fingerprint"],
            field="new_training_data_fingerprint",
        )
        if row["training_data_changed"] != (old != new):
            raise ValueError("calibrator training-data change flag mismatch")
        _validate_fingerprint(row, field="change_fingerprint")


def validate_reference_model_rebuild_plan(frame: pl.DataFrame) -> None:
    if frame.schema != reference_model_rebuild_plan_schema():
        raise ValueError("reference model rebuild plan schema mismatch")
    required = tuple(
        field
        for field in reference_model_rebuild_plan_schema()
        if field
        not in {
            "old_training_data_fingerprint",
            "new_training_data_fingerprint",
            "training_data_changed",
        }
    )
    _reject_nulls(frame, required)
    if not frame.equals(frame.sort("artifact_id")):
        raise ValueError("reference model rebuild plan is not sorted")
    if frame["artifact_id"].n_unique() != frame.height:
        raise ValueError("reference model rebuild plan repeats an artifact")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != REFERENCE_MODEL_REBUILD_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported reference model rebuild plan version")
        artifact_type = _artifact_type(row["artifact_type"])
        if row["source_impact_status"] not in {"affected", "reusable_as_is"}:
            raise ValueError("unsupported source impact status")
        affected = row["source_impact_status"] == "affected"
        training = _training_evidence_from_plan_row(row)
        if artifact_type != "calibrator" and training is not None:
            raise ValueError("only calibrators may carry training-data evidence")
        expected_rebuild, expected_action = _rebuild_decision(
            row,
            training=training,
        )
        if row["rebuild_status"] != ("rebuild" if expected_rebuild else "reuse"):
            raise ValueError("reference model rebuild status mismatch")
        if row["rebuild_action"] != expected_action:
            raise ValueError("reference model rebuild action mismatch")
        if affected and artifact_type in {
            "reference_prototype",
            "regional_prototype",
            "candidate_set",
            "classifier",
        }:
            if not row["affected_species"] or not row["affected_routes"]:
                raise ValueError("affected model lacks species or route scope")
        if affected and artifact_type in {
            "regional_prototype",
            "candidate_set",
        } and not row["affected_regions"]:
            raise ValueError("affected regional artifact lacks region scope")
        for field in (
            "revision_fingerprint",
            "current_artifact_fingerprint",
            "source_impact_fingerprint",
        ):
            _sha256(row[field], field=field)
        _validate_fingerprint(row, field="plan_fingerprint")


def write_reference_model_rebuild_plan(
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    validate_reference_model_rebuild_plan(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= REFERENCE_MODEL_REBUILD_PLAN_FILE
    return write_parquet(frame, destination)


def _rebuild_decision(
    impact: Mapping[str, object],
    *,
    training: Mapping[str, object] | None,
) -> tuple[bool, str]:
    artifact_type = _artifact_type(impact["artifact_type"])
    impact_status = (
        impact["source_impact_status"]
        if "source_impact_status" in impact
        else impact["impact_status"]
    )
    if impact_status == "reusable_as_is":
        return False, "reuse_unchanged_artifact"
    if artifact_type == "calibrator":
        if training is None:
            raise ValueError("affected calibrator lacks training-data evidence")
        if not bool(training["training_data_changed"]):
            return False, "reuse_calibrator_training_data_unchanged"
        return True, "rebuild_calibrator_training_data_changed"
    return True, {
        "reference_prototype": "rebuild_affected_species_prototypes",
        "regional_prototype": "rebuild_affected_regional_prototypes",
        "candidate_set": "rebuild_affected_regional_candidate_index",
        "classifier": "rebuild_affected_classifier_rows_and_model",
    }[artifact_type]


def _training_evidence_from_plan_row(
    row: Mapping[str, object],
) -> Mapping[str, object] | None:
    values = (
        row["old_training_data_fingerprint"],
        row["new_training_data_fingerprint"],
        row["training_data_changed"],
    )
    if values == (None, None, None):
        return None
    if any(value is None for value in values):
        raise ValueError("partial calibrator training-data evidence")
    old = _sha256(values[0], field="old_training_data_fingerprint")
    new = _sha256(values[1], field="new_training_data_fingerprint")
    if values[2] != (old != new):
        raise ValueError("calibrator training-data change flag mismatch")
    return {"training_data_changed": values[2]}


def _artifact_type(value: object) -> str:
    artifact_type = _required_text(value, field="artifact_type")
    if artifact_type not in MODEL_ARTIFACT_TYPES:
        raise ValueError(f"unsupported reference model artifact: {artifact_type!r}")
    return artifact_type


def _validate_fingerprint(row: Mapping[str, object], *, field: str) -> None:
    payload = dict(row)
    fingerprint = payload.pop(field)
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError(f"{field} mismatch")


def _reject_nulls(frame: pl.DataFrame, fields: tuple[str, ...]) -> None:
    if any(frame[field].null_count() for field in fields):
        raise ValueError("reference model artifacts contain null required fields")


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
    "CALIBRATOR_TRAINING_CHANGE_SCHEMA",
    "CALIBRATOR_TRAINING_CHANGE_SCHEMA_VERSION",
    "MODEL_ARTIFACT_TYPES",
    "REFERENCE_MODEL_REBUILD_PLAN_FILE",
    "REFERENCE_MODEL_REBUILD_PLAN_SCHEMA_VERSION",
    "calculate_reference_model_rebuild_plan",
    "calibrator_training_change_frame",
    "reference_model_artifact_ids_to_rebuild",
    "reference_model_rebuild_metrics",
    "reference_model_rebuild_plan_schema",
    "validate_calibrator_training_changes",
    "validate_reference_model_rebuild_plan",
    "write_reference_model_rebuild_plan",
]
