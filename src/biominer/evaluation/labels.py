from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from biominer.references.schemas import (
    REFERENCE_LIFE_STAGES,
    REFERENCE_ROUTES,
    REFERENCE_VIEWS,
    REFERENCE_VISUAL_DOMAINS,
)


REVIEWED_LABEL_V1_SCHEMA_VERSION = "reviewed-labels-v1"
REVIEWED_LABEL_SCHEMA_VERSION = "reviewed-labels-v2"

LABEL_LEVELS = frozenset({"photo", "object", "family", "species", "negative"})
REVIEW_CONFIDENCE_VALUES = frozenset({"high", "medium", "low", "unknown"})
LABEL_CERTAINTY_VALUES = REVIEW_CONFIDENCE_VALUES
SOURCE_QUERY_TIER_VALUES = frozenset({"T1", "T2", "T3", "T4", "T5"})
SECOND_REVIEW_STATUS_VALUES = frozenset(
    {
        "not_required",
        "pending",
        "second_review_required",
        "completed",
        "conflict",
        "unknown",
    }
)

REVIEWED_LABEL_V1_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "detection_id": pl.String,
    "crop_hash": pl.String,
    "label_level": pl.String,
    "is_butterfly": pl.Boolean,
    "accepted_taxon_key": pl.String,
    "scientific_name": pl.String,
    "family_key": pl.String,
    "family": pl.String,
    "genus_key": pl.String,
    "genus": pl.String,
    "label_source": pl.String,
    "reviewer_id": pl.String,
    "reviewed_at": pl.String,
    "review_confidence": pl.String,
    "review_notes": pl.String,
}

REVIEWED_LABEL_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    **REVIEWED_LABEL_V1_SCHEMA,
    "target_present": pl.Boolean,
    "label_certainty": pl.String,
    "life_stage": pl.String,
    "visual_domain": pl.String,
    "view": pl.String,
    "route": pl.String,
    "geo_cluster_id": pl.String,
    "source_query_tier": pl.String,
    "source_query_term": pl.String,
    "duplicate_group_id": pl.String,
    "observer_owner_group_id": pl.String,
    "dataset_split": pl.String,
    "second_review_status": pl.String,
    "ambiguity_reason": pl.String,
    "unsuitable_for_species_identification": pl.Boolean,
}


def empty_reviewed_label_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=REVIEWED_LABEL_SCHEMA)


def normalize_reviewed_label_frame(
    frame: pl.DataFrame,
    *,
    target_accepted_taxon_key: str | None = None,
) -> pl.DataFrame:
    """Return a deterministic v2 frame, migrating complete v1 inputs only."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("reviewed labels must be a Polars DataFrame")
    columns = set(frame.columns)
    v2_columns = set(REVIEWED_LABEL_SCHEMA)
    v1_columns = set(REVIEWED_LABEL_V1_SCHEMA)
    v2_only_columns = v2_columns - v1_columns
    if v2_columns <= columns:
        versions = set(frame["schema_version"].to_list())
        if frame.height and versions != {REVIEWED_LABEL_SCHEMA_VERSION}:
            raise ValueError(
                "unsupported reviewed-label schema_version: "
                + ", ".join(sorted(str(value) for value in versions))
            )
        return _canonical_reviewed_label_order(frame)
    present_v2_only = sorted(v2_only_columns & columns)
    if present_v2_only == ["schema_version"]:
        versions = set(frame["schema_version"].to_list())
        if not frame.height or versions == {REVIEWED_LABEL_V1_SCHEMA_VERSION}:
            return migrate_v1_reviewed_label_frame(
                frame,
                target_accepted_taxon_key=target_accepted_taxon_key,
            )
        raise ValueError(
            "unsupported reviewed-label schema_version: "
            + ", ".join(sorted(str(value) for value in versions))
        )
    if present_v2_only:
        missing = sorted(v2_columns - columns)
        raise ValueError(
            "incomplete reviewed-label v2 frame: "
            f"present_v2_columns={present_v2_only}, missing_columns={missing}"
        )
    if v1_columns <= columns:
        return migrate_v1_reviewed_label_frame(
            frame,
            target_accepted_taxon_key=target_accepted_taxon_key,
        )
    missing_v1 = sorted(v1_columns - columns)
    raise ValueError(
        "reviewed labels do not match v1 or v2: "
        f"missing_v1_columns={missing_v1}"
    )


def migrate_v1_reviewed_label_frame(
    frame: pl.DataFrame,
    *,
    target_accepted_taxon_key: str | None = None,
) -> pl.DataFrame:
    """Migrate v1 labels without inventing unavailable target provenance."""

    source = frame
    if "schema_version" in source.columns:
        versions = set(source["schema_version"].to_list())
        if source.height and versions != {REVIEWED_LABEL_V1_SCHEMA_VERSION}:
            raise ValueError(
                "reviewed-label v1 migration requires schema_version "
                f"{REVIEWED_LABEL_V1_SCHEMA_VERSION!r}"
            )
        source = source.drop("schema_version")
    missing = sorted(set(REVIEWED_LABEL_V1_SCHEMA) - set(source.columns))
    if missing:
        raise ValueError(
            f"reviewed-label v1 migration is missing columns: {missing}"
        )
    partial_v2 = sorted(
        (set(REVIEWED_LABEL_SCHEMA) - set(REVIEWED_LABEL_V1_SCHEMA))
        & set(source.columns)
    )
    if partial_v2:
        raise ValueError(
            "reviewed-label v1 migration received v2-only columns: "
            f"{partial_v2}"
        )
    invalid_dtypes = _schema_dtype_mismatches(source, REVIEWED_LABEL_V1_SCHEMA)
    if invalid_dtypes:
        raise ValueError(
            "reviewed-label v1 migration has incompatible column types: "
            f"{invalid_dtypes}"
        )
    target_key = _text(target_accepted_taxon_key)
    if target_accepted_taxon_key is not None and not target_key:
        raise ValueError("target_accepted_taxon_key cannot be blank")
    species_label = pl.col("label_level").cast(pl.String) == "species"
    butterfly_positive = pl.col("is_butterfly").cast(pl.Boolean)
    non_butterfly = butterfly_positive.not_()
    accepted_key = pl.col("accepted_taxon_key").cast(pl.String)
    scientific_name = pl.col("scientific_name").cast(pl.String)
    accepted_key_present = accepted_key.str.strip_chars() != ""
    scientific_name_present = scientific_name.str.strip_chars() != ""
    if target_key:
        target_present = (
            pl.when(non_butterfly)
            .then(pl.lit(False))
            .when(
                butterfly_positive
                & species_label
                & accepted_key_present
            )
            .then(accepted_key == target_key)
            .otherwise(pl.lit(None, dtype=pl.Boolean))
        )
    else:
        target_present = (
            pl.when(non_butterfly)
            .then(pl.lit(False))
            .otherwise(pl.lit(None, dtype=pl.Boolean))
        )
    species_suitability = (
        pl.when(
            butterfly_positive
            & species_label
            & accepted_key_present
            & scientific_name_present
        )
        .then(pl.lit(False))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
    )
    migrated = source.with_columns(
        pl.lit(REVIEWED_LABEL_SCHEMA_VERSION).alias("schema_version"),
        target_present.alias("target_present"),
        pl.col("review_confidence").alias("label_certainty"),
        pl.lit("unknown").alias("life_stage"),
        pl.lit("ambiguous").alias("visual_domain"),
        pl.lit("unknown").alias("view"),
        pl.lit(None, dtype=pl.String).alias("route"),
        pl.lit(None, dtype=pl.String).alias("geo_cluster_id"),
        pl.lit(None, dtype=pl.String).alias("source_query_tier"),
        pl.lit(None, dtype=pl.String).alias("source_query_term"),
        pl.lit(None, dtype=pl.String).alias("duplicate_group_id"),
        pl.lit(None, dtype=pl.String).alias("observer_owner_group_id"),
        pl.lit("unassigned").alias("dataset_split"),
        pl.lit("unknown").alias("second_review_status"),
        pl.lit("legacy_v1_missing_target_aware_fields").alias(
            "ambiguity_reason"
        ),
        species_suitability.alias("unsuitable_for_species_identification"),
    )
    return _canonical_reviewed_label_order(migrated)


def validate_reviewed_label_frame(frame: pl.DataFrame) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    missing = [column for column in REVIEWED_LABEL_SCHEMA if column not in frame.columns]
    if missing:
        findings.append(
            _finding(
                "fatal",
                "missing_required_columns",
                "reviewed labels are missing required columns",
                {"columns": missing},
            )
        )
        return findings

    dtype_mismatches = _schema_dtype_mismatches(frame, REVIEWED_LABEL_SCHEMA)
    if dtype_mismatches:
        findings.append(
            _finding(
                "fatal",
                "invalid_column_types",
                "reviewed labels contain incompatible column types",
                {"columns": dtype_mismatches},
            )
        )
    rows = _rows(frame)
    _append_invalid_schema_version_findings(findings, rows)
    _append_invalid_label_level_findings(findings, rows)
    _append_invalid_confidence_findings(findings, rows)
    _append_invalid_target_aware_findings(findings, rows)
    _append_missing_butterfly_taxonomy_findings(findings, rows)
    _append_duplicate_species_conflict_findings(findings, rows)
    _append_review_metadata_warnings(findings, rows)
    _append_photo_without_object_warning(findings, rows)
    return findings


def read_reviewed_labels(
    path: str | Path,
    *,
    target_accepted_taxon_key: str | None = None,
) -> pl.DataFrame:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix == ".parquet":
        frame = pl.read_parquet(source)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pl.read_ndjson(source)
    elif suffix == ".json":
        frame = pl.read_json(source)
    else:
        raise ValueError(
            f"unsupported reviewed-label format: {suffix or '<none>'}"
        )
    return normalize_reviewed_label_frame(
        frame,
        target_accepted_taxon_key=target_accepted_taxon_key,
    )


def _canonical_reviewed_label_order(frame: pl.DataFrame) -> pl.DataFrame:
    extras = sorted(set(frame.columns) - set(REVIEWED_LABEL_SCHEMA))
    return frame.select([*REVIEWED_LABEL_SCHEMA, *extras])


def _schema_dtype_mismatches(
    frame: pl.DataFrame,
    schema: dict[str, pl.DataType],
) -> dict[str, dict[str, str]]:
    mismatches: dict[str, dict[str, str]] = {}
    for column, expected in schema.items():
        if column not in frame.columns:
            continue
        actual = frame.schema[column]
        if actual not in {expected, pl.Null}:
            mismatches[column] = {
                "expected": str(expected),
                "actual": str(actual),
            }
    return mismatches


def _rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    return frame.select(list(REVIEWED_LABEL_SCHEMA)).to_dicts()


def _append_invalid_schema_version_findings(
    findings: list[dict[str, object]],
    rows: list[dict[str, Any]],
) -> None:
    invalid = sorted(
        {
            _text(row.get("schema_version"))
            for row in rows
            if _text(row.get("schema_version")) != REVIEWED_LABEL_SCHEMA_VERSION
        }
    )
    if invalid:
        findings.append(
            _finding(
                "fatal",
                "invalid_schema_version",
                "reviewed labels contain an unsupported schema_version",
                {
                    "values": invalid,
                    "expected": REVIEWED_LABEL_SCHEMA_VERSION,
                },
            )
        )


def _append_invalid_target_aware_findings(
    findings: list[dict[str, object]],
    rows: list[dict[str, Any]],
) -> None:
    _append_invalid_choice_finding(
        findings,
        rows,
        field="label_certainty",
        allowed=LABEL_CERTAINTY_VALUES,
        code="invalid_label_certainty",
    )
    _append_invalid_choice_finding(
        findings,
        rows,
        field="life_stage",
        allowed=REFERENCE_LIFE_STAGES,
        code="invalid_life_stage",
    )
    _append_invalid_choice_finding(
        findings,
        rows,
        field="visual_domain",
        allowed=REFERENCE_VISUAL_DOMAINS,
        code="invalid_visual_domain",
    )
    _append_invalid_choice_finding(
        findings,
        rows,
        field="view",
        allowed=REFERENCE_VIEWS,
        code="invalid_view",
    )
    _append_invalid_choice_finding(
        findings,
        rows,
        field="route",
        allowed=REFERENCE_ROUTES,
        code="invalid_route",
        allow_missing=True,
    )
    _append_invalid_choice_finding(
        findings,
        rows,
        field="source_query_tier",
        allowed=SOURCE_QUERY_TIER_VALUES,
        code="invalid_source_query_tier",
        allow_missing=True,
    )
    _append_invalid_choice_finding(
        findings,
        rows,
        field="second_review_status",
        allowed=SECOND_REVIEW_STATUS_VALUES,
        code="invalid_second_review_status",
    )
    invalid_target_values = [
        _row_ref(index, row)
        for index, row in enumerate(rows)
        if row.get("target_present") is not None
        and not isinstance(row.get("target_present"), bool)
    ]
    if invalid_target_values:
        findings.append(
            _finding(
                "fatal",
                "invalid_target_present",
                "target_present must be Boolean or null for migrated v1 labels",
                {"rows": invalid_target_values},
            )
        )
    target_missing_taxonomy = [
        _row_ref(index, row)
        for index, row in enumerate(rows)
        if row.get("target_present") is True
        and (
            not _text(row.get("accepted_taxon_key"))
            or not _text(row.get("scientific_name"))
        )
    ]
    if target_missing_taxonomy:
        findings.append(
            _finding(
                "fatal",
                "target_label_missing_taxonomy",
                "target-present labels require accepted_taxon_key and scientific_name",
                {"rows": target_missing_taxonomy},
            )
        )
    invalid_unsuitable = [
        _row_ref(index, row)
        for index, row in enumerate(rows)
        if row.get("unsuitable_for_species_identification") is not None
        and not isinstance(
            row.get("unsuitable_for_species_identification"), bool
        )
    ]
    if invalid_unsuitable:
        findings.append(
            _finding(
                "fatal",
                "invalid_species_identification_suitability",
                "unsuitable_for_species_identification must be Boolean or null",
                {"rows": invalid_unsuitable},
            )
        )
    unsuitable_without_reason = [
        _row_ref(index, row)
        for index, row in enumerate(rows)
        if row.get("unsuitable_for_species_identification") is True
        and not _text(row.get("ambiguity_reason"))
    ]
    if unsuitable_without_reason:
        findings.append(
            _finding(
                "fatal",
                "unsuitable_label_missing_ambiguity_reason",
                "labels unsuitable for species identification require an ambiguity_reason",
                {"rows": unsuitable_without_reason},
            )
        )
    route_conflicts = [
        {
            **_row_ref(index, row),
            "route": _text(row.get("route")),
            "life_stage": _text(row.get("life_stage")),
            "visual_domain": _text(row.get("visual_domain")),
        }
        for index, row in enumerate(rows)
        if _route_conflicts_with_dimensions(row)
    ]
    if route_conflicts:
        findings.append(
            _finding(
                "fatal",
                "route_dimension_conflict",
                "reviewed-label route conflicts with life_stage or visual_domain",
                {"rows": route_conflicts},
            )
        )


def _append_invalid_choice_finding(
    findings: list[dict[str, object]],
    rows: list[dict[str, Any]],
    *,
    field: str,
    allowed: frozenset[str],
    code: str,
    allow_missing: bool = False,
) -> None:
    invalid = sorted(
        {
            _text(row.get(field))
            for row in rows
            if (
                (not allow_missing or _text(row.get(field)))
                and _text(row.get(field)) not in allowed
            )
        }
    )
    if invalid:
        findings.append(
            _finding(
                "fatal",
                code,
                f"reviewed labels contain invalid {field} values",
                {"values": invalid, "allowed": sorted(allowed)},
            )
        )


def _route_conflicts_with_dimensions(row: dict[str, Any]) -> bool:
    route = _text(row.get("route"))
    if not route:
        return False
    expected = {
        "adult_field": ("adult", "live_field"),
        "larval": ("larva", "live_field"),
        "pupal": ("pupa", "live_field"),
        "egg": ("egg", "live_field"),
        "pinned_specimen": (None, "pinned_specimen"),
    }.get(route)
    if expected is None:
        return False
    life_stage, visual_domain = expected
    return (
        (life_stage is not None and _text(row.get("life_stage")) != life_stage)
        or _text(row.get("visual_domain")) != visual_domain
    )


def _append_invalid_label_level_findings(findings: list[dict[str, object]], rows: list[dict[str, Any]]) -> None:
    invalid = sorted(
        {
            str(row.get("label_level") or "").strip()
            for row in rows
            if str(row.get("label_level") or "").strip() not in LABEL_LEVELS
        }
    )
    if invalid:
        findings.append(
            _finding(
                "fatal",
                "invalid_label_level",
                "reviewed labels contain invalid label_level values",
                {"values": invalid, "allowed": sorted(LABEL_LEVELS)},
            )
        )


def _append_invalid_confidence_findings(findings: list[dict[str, object]], rows: list[dict[str, Any]]) -> None:
    invalid = sorted(
        {
            str(row.get("review_confidence") or "").strip()
            for row in rows
            if str(row.get("review_confidence") or "").strip() not in REVIEW_CONFIDENCE_VALUES
        }
    )
    if invalid:
        findings.append(
            _finding(
                "fatal",
                "invalid_review_confidence",
                "reviewed labels contain invalid review_confidence values",
                {"values": invalid, "allowed": sorted(REVIEW_CONFIDENCE_VALUES)},
            )
        )


def _append_missing_butterfly_taxonomy_findings(findings: list[dict[str, object]], rows: list[dict[str, Any]]) -> None:
    bad_rows = [
        _row_ref(index, row)
        for index, row in enumerate(rows)
        if bool(row.get("is_butterfly")) and (not _text(row.get("family")) or not _text(row.get("scientific_name")))
    ]
    if bad_rows:
        findings.append(
            _finding(
                "fatal",
                "butterfly_label_missing_taxonomy",
                "butterfly-positive reviewed labels require family and scientific_name",
                {"rows": bad_rows},
            )
        )


def _append_duplicate_species_conflict_findings(findings: list[dict[str, object]], rows: list[dict[str, Any]]) -> None:
    labels_by_object: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    for row in rows:
        detection_id = _text(row.get("detection_id"))
        if not detection_id or _text(row.get("label_level")) not in {"object", "family", "species"}:
            continue
        if not bool(row.get("is_butterfly")):
            continue
        signature = (_text(row.get("accepted_taxon_key")), _text(row.get("scientific_name")))
        if not any(signature):
            continue
        key = (_text(row.get("source")), _text(row.get("flickr_photo_id")), detection_id)
        labels_by_object.setdefault(key, set()).add(signature)

    conflicts = [
        {"source": key[0], "flickr_photo_id": key[1], "detection_id": key[2], "labels": sorted(values)}
        for key, values in sorted(labels_by_object.items())
        if len(values) > 1
    ]
    if conflicts:
        findings.append(
            _finding(
                "fatal",
                "duplicate_object_conflicting_species_labels",
                "duplicate object-level reviewed labels contain conflicting species labels",
                {"conflicts": conflicts},
            )
        )


def _append_review_metadata_warnings(findings: list[dict[str, object]], rows: list[dict[str, Any]]) -> None:
    missing_reviewer = [_row_ref(index, row) for index, row in enumerate(rows) if not _text(row.get("reviewer_id"))]
    if missing_reviewer:
        findings.append(
            _finding(
                "warning",
                "missing_reviewer_id",
                "one or more reviewed labels is missing reviewer_id",
                {"rows": missing_reviewer},
            )
        )

    missing_reviewed_at = [_row_ref(index, row) for index, row in enumerate(rows) if not _text(row.get("reviewed_at"))]
    if missing_reviewed_at:
        findings.append(
            _finding(
                "warning",
                "missing_reviewed_at",
                "one or more reviewed labels is missing reviewed_at",
                {"rows": missing_reviewed_at},
            )
        )

    low_confidence = [
        _row_ref(index, row)
        for index, row in enumerate(rows)
        if _text(row.get("review_confidence")) == "low"
    ]
    if low_confidence:
        findings.append(
            _finding(
                "warning",
                "low_confidence_labels",
                "one or more reviewed labels has low confidence",
                {"rows": low_confidence},
            )
        )


def _append_photo_without_object_warning(findings: list[dict[str, object]], rows: list[dict[str, Any]]) -> None:
    object_keys = {
        (_text(row.get("source")), _text(row.get("flickr_photo_id")))
        for row in rows
        if (
            _text(row.get("label_level")) in {"object", "family", "species", "negative"}
            and _text(row.get("detection_id"))
        )
    }
    photo_only = [
        _row_ref(index, row)
        for index, row in enumerate(rows)
        if (
            _text(row.get("label_level")) == "photo"
            and (_text(row.get("source")), _text(row.get("flickr_photo_id"))) not in object_keys
        )
    ]
    if photo_only:
        findings.append(
            _finding(
                "warning",
                "photo_label_without_object_label",
                "one or more photo-level labels has no matching object-level label",
                {"rows": photo_only},
            )
        )


def _row_ref(index: int, row: dict[str, Any]) -> dict[str, object]:
    return {
        "row_index": index,
        "source": _text(row.get("source")),
        "flickr_photo_id": _text(row.get("flickr_photo_id")),
        "detection_id": _text(row.get("detection_id")),
    }


def _finding(severity: str, code: str, message: str, details: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "details": dict(details or {}),
    }


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


__all__ = [
    "LABEL_CERTAINTY_VALUES",
    "LABEL_LEVELS",
    "REVIEWED_LABEL_SCHEMA",
    "REVIEWED_LABEL_SCHEMA_VERSION",
    "REVIEWED_LABEL_V1_SCHEMA",
    "REVIEWED_LABEL_V1_SCHEMA_VERSION",
    "REVIEW_CONFIDENCE_VALUES",
    "SECOND_REVIEW_STATUS_VALUES",
    "SOURCE_QUERY_TIER_VALUES",
    "empty_reviewed_label_frame",
    "migrate_v1_reviewed_label_frame",
    "normalize_reviewed_label_frame",
    "read_reviewed_labels",
    "validate_reviewed_label_frame",
]
