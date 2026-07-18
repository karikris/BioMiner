"""Source-bound ButterflyLens geographic candidate-evidence export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.integration.butterflylens_model_export import (
    ButterflyLensModelLayer,
    validate_butterflylens_model_layer,
)
from biominer.integration.butterflylens_pool_handoff import (
    BUTTERFLYLENS_ROLE_DEFAULTS,
)
from biominer.integration.product_handoff import (
    normalize_product_artifacts,
    validate_fingerprint,
    validate_git_sha,
)
from biominer.storage.content_address import sha256_file
from biominer.storage.parquet import write_parquet


BUTTERFLYLENS_GEOGRAPHIC_IMPACT_VERSION = (
    "biominer-butterflylens-geographic-impact-v1.0.0"
)
BUTTERFLYLENS_GEOGRAPHIC_IMPACT_FILE = "butterflylens_geographic_impact_cells.parquet"
BUTTERFLYLENS_TARGET_CELL_VERSION = "butterflylens-geographic-impact-cell:v1.0.0"
_INPUT_FIELDS = frozenset(
    {
        "flickr_record_id",
        "geography_availability",
        "h3_cell",
        "h3_version",
        "h3_resolution",
        "source_precision_metres",
        "published_h3_resolution",
        "public_geometry_status",
        "public_geometry_reason",
        "latest_flickr_event_date",
        "geographic_evidence_fingerprint",
    }
)
_GEOGRAPHY_STATES = frozenset({"h3", "no_geo", "unassigned_geo", "withheld"})
_PUBLIC_GEOMETRY_STATES = frozenset({"available", "generalized", "withheld"})
_COUNT_NAMES = (
    "ala_baseline",
    "flickr_candidate",
    "yoloe_butterfly",
    "bioclip_species_candidate",
    "community_reviewed",
    "human_supported",
    "release_ready",
)
_IMPACT_NAMES = (
    "potential_coverage_gap",
    "human_supported_additional",
    "release_ready_additional",
)
_H3_PATTERN = re.compile(r"[0-9a-f]{15}\Z")


def _schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "schema_version": pl.String,
        "target_cell_schema_version": pl.String,
        "project_id": pl.String,
        "run_id": pl.String,
        "geographic_impact_id": pl.String,
        "snapshot_mode": pl.String,
        "source_commit": pl.String,
        "geography_availability": pl.String,
        "cell_id": pl.String,
        "grid": pl.String,
        "h3_version": pl.String,
        "h3_resolution": pl.UInt8,
        "accepted_taxon_key": pl.String,
        "source_snapshot_fingerprint": pl.String,
        "flickr_snapshot_fingerprint": pl.String,
        "ala_snapshot_fingerprint": pl.String,
        "provider_union_fingerprint": pl.String,
        "review_projection_fingerprint": pl.String,
        "quality_snapshot_fingerprint": pl.String,
    }
    for name in _COUNT_NAMES:
        schema[f"{name}_count"] = pl.UInt64
        schema[f"{name}_count_state"] = pl.String
        schema[f"{name}_count_reason"] = pl.String
    for name in _IMPACT_NAMES:
        schema[name] = pl.Boolean
        schema[f"{name}_state"] = pl.String
        schema[f"{name}_reason"] = pl.String
    schema.update(
        {
            "nearest_ala_evidence_distance_m": pl.Float64,
            "nearest_ala_evidence_distance_state": pl.String,
            "nearest_ala_evidence_distance_reason": pl.String,
            "latest_ala_event_date": pl.String,
            "latest_ala_event_date_state": pl.String,
            "latest_ala_event_date_reason": pl.String,
            "latest_flickr_event_date": pl.String,
            "latest_flickr_event_date_state": pl.String,
            "latest_flickr_event_date_reason": pl.String,
            "data_deficiency_state": pl.String,
            "data_deficiency_reason": pl.String,
            "public_geometry_status": pl.String,
            "source_precision_metres": pl.Float64,
            "published_h3_resolution": pl.UInt8,
            "public_geometry_reason": pl.String,
            "evidence_fingerprints": pl.List(pl.String),
            "candidate_only_is_occurrence": pl.Boolean,
            "no_geo_is_biological_absence": pl.Boolean,
            "database_primary_key_included": pl.Boolean,
            "scientific_claim_allowed": pl.Boolean,
            "impact_fingerprint": pl.String,
        }
    )
    return schema


BUTTERFLYLENS_GEOGRAPHIC_IMPACT_SCHEMA = _schema()


@dataclass(frozen=True, slots=True)
class ButterflyLensGeographicExport:
    root: Path
    path: Path
    artifact: dict[str, object]


def build_butterflylens_geographic_impact(
    *,
    model_layer: ButterflyLensModelLayer,
    geographic_records: Sequence[Mapping[str, object]],
    source_commit: str,
) -> pl.DataFrame:
    """Project model candidates into H3 evidence or explicit no-geo exclusions."""

    validate_butterflylens_model_layer(model_layer)
    commit = validate_git_sha(source_commit, field="source_commit")
    inputs = _normalize_inputs(geographic_records)
    sources = {
        str(row["flickr_record_id"]): row
        for row in model_layer.flickr_source_records.iter_rows(named=True)
    }
    if set(inputs) != set(sources):
        raise ValueError("ButterflyLens geographic source identities differ")
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for evidence in model_layer.model_evidence.iter_rows(named=True):
        source_id = str(evidence["flickr_record_id"])
        geographic = inputs[source_id]
        key = (
            geographic["geography_availability"],
            geographic["h3_cell"],
            geographic["h3_version"],
            geographic["h3_resolution"],
            evidence["candidate_accepted_taxon_key"],
            geographic["public_geometry_status"],
            geographic["source_precision_metres"],
            geographic["published_h3_resolution"],
            geographic["public_geometry_reason"],
        )
        grouped.setdefault(key, []).append(
            {"source": sources[source_id], "evidence": evidence, "geo": geographic}
        )
    rows = [
        _build_row(items, source_commit=commit)
        for _, items in sorted(
            grouped.items(), key=lambda item: tuple(str(v) for v in item[0])
        )
    ]
    frame = pl.DataFrame(
        rows, schema=BUTTERFLYLENS_GEOGRAPHIC_IMPACT_SCHEMA, strict=True
    ).sort(
        "geography_availability",
        "cell_id",
        "accepted_taxon_key",
        "geographic_impact_id",
        nulls_last=True,
    )
    validate_butterflylens_geographic_impact(frame)
    return frame


def validate_butterflylens_geographic_impact(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    if frame.schema != BUTTERFLYLENS_GEOGRAPHIC_IMPACT_SCHEMA or frame.is_empty():
        raise ValueError("ButterflyLens geographic impact schema or rows differ")
    if not frame.equals(
        frame.sort(
            "geography_availability",
            "cell_id",
            "accepted_taxon_key",
            "geographic_impact_id",
            nulls_last=True,
        )
    ):
        raise ValueError("ButterflyLens geographic impact rows are not canonical")
    if frame["impact_fingerprint"].n_unique() != frame.height:
        raise ValueError("ButterflyLens geographic impact fingerprints repeat")
    scope = frame.select("project_id", "run_id", "source_commit").unique()
    if scope.height != 1:
        raise ValueError("ButterflyLens geographic impact scope differs")
    for row in frame.iter_rows(named=True):
        _validate_row(row)


def butterflylens_geographic_cell_documents(
    frame: pl.DataFrame,
) -> list[dict[str, object]]:
    """Translate located rows to the exact pinned ButterflyLens cell wire shape."""

    validate_butterflylens_geographic_impact(frame)
    documents: list[dict[str, object]] = []
    for row in frame.filter(pl.col("geography_availability") == "h3").iter_rows(
        named=True
    ):
        documents.append(
            {
                "schema_version": BUTTERFLYLENS_TARGET_CELL_VERSION,
                "cell_id": row["cell_id"],
                "grid": row["grid"],
                "h3_version": row["h3_version"],
                "h3_resolution": row["h3_resolution"],
                "project_id": row["project_id"],
                "run_id": row["run_id"],
                "snapshot_mode": row["snapshot_mode"],
                "accepted_taxon_key": row["accepted_taxon_key"],
                "ala_snapshot_fingerprint": _wire_sha(row["ala_snapshot_fingerprint"]),
                "flickr_snapshot_fingerprint": _wire_sha(
                    row["flickr_snapshot_fingerprint"]
                ),
                "provider_union_fingerprint": _wire_sha(
                    row["provider_union_fingerprint"]
                ),
                "review_projection_fingerprint": _wire_sha(
                    row["review_projection_fingerprint"]
                ),
                "quality_snapshot_fingerprint": _wire_sha(
                    row["quality_snapshot_fingerprint"]
                ),
                "counts": {
                    name: {
                        "status": row[f"{name}_count_state"],
                        "value": row[f"{name}_count"],
                        "reason": row[f"{name}_count_reason"],
                    }
                    for name in _COUNT_NAMES
                },
                "impact": {
                    name: {
                        "status": row[f"{name}_state"],
                        "value": row[name],
                        "reason": row[f"{name}_reason"],
                    }
                    for name in _IMPACT_NAMES
                },
                "nearest_ala_evidence_distance": {
                    "status": row["nearest_ala_evidence_distance_state"],
                    "metres": row["nearest_ala_evidence_distance_m"],
                    "reason": row["nearest_ala_evidence_distance_reason"],
                },
                "latest_ala_event_date": row["latest_ala_event_date"],
                "latest_flickr_event_date": row["latest_flickr_event_date"],
                "data_deficiency_state": row["data_deficiency_state"],
                "public_geometry": {
                    "status": row["public_geometry_status"],
                    "source_precision_metres": row["source_precision_metres"],
                    "published_h3_resolution": row["published_h3_resolution"],
                    "reason": row["public_geometry_reason"],
                },
                "evidence_fingerprints": [
                    _wire_sha(value) for value in row["evidence_fingerprints"]
                ],
                "cell_fingerprint": _wire_sha(row["impact_fingerprint"]),
                "scientific_claim_allowed": False,
            }
        )
    return documents


def export_butterflylens_geographic_impact(
    *, frame: pl.DataFrame, output_root: str | Path
) -> ButterflyLensGeographicExport:
    """Create and validate the immutable geographic role artifact."""

    validate_butterflylens_geographic_impact(frame)
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("ButterflyLens handoff root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    directory = root / "artifacts" / "geographic"
    if directory.exists():
        raise FileExistsError(
            f"ButterflyLens geographic directory is create-only: {directory}"
        )
    directory.mkdir(parents=True)
    path = write_parquet(
        frame, directory / BUTTERFLYLENS_GEOGRAPHIC_IMPACT_FILE, overwrite=False
    )
    artifact = _artifact(path, frame)
    result = ButterflyLensGeographicExport(root=root, path=path, artifact=artifact)
    validate_butterflylens_geographic_export(result.root, result.artifact)
    return result


def validate_butterflylens_geographic_export(
    root: str | Path, artifact: Mapping[str, object]
) -> None:
    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError("ButterflyLens handoff root must not be a symlink")
    normalized = normalize_product_artifacts(
        [artifact],
        required_roles=("geographic_impact",),
        producer_repository="karikris/BioMiner",
        producer_commit="0" * 40,
    )[0]
    filename, schema_version = BUTTERFLYLENS_ROLE_DEFAULTS["geographic_impact"]
    expected_relative = f"artifacts/geographic/{filename}"
    if (
        normalized["availability"] != "available"
        or normalized["schema_version"] != schema_version
        or normalized["relative_path"] != expected_relative
        or normalized["evidence_maturity_label"] is not None
    ):
        raise ValueError("ButterflyLens geographic artifact contract differs")
    directory = root_path / "artifacts" / "geographic"
    path = root_path / expected_relative
    if directory.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("ButterflyLens geographic artifact is unavailable")
    if set(directory.iterdir()) != {path}:
        raise ValueError("ButterflyLens geographic directory file set differs")
    if (
        path.stat().st_size != normalized["byte_count"]
        or sha256_file(path) != normalized["sha256"]
    ):
        raise ValueError("ButterflyLens geographic physical identity differs")
    frame = pl.read_parquet(path)
    validate_butterflylens_geographic_impact(frame)
    if (
        frame.height != normalized["row_count"]
        or _semantic_fingerprint(frame) != normalized["semantic_fingerprint"]
        or _parent_fingerprints(frame) != normalized["parent_fingerprints"]
    ):
        raise ValueError("ButterflyLens geographic semantic identity differs")


def _normalize_inputs(
    records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
    ):
        raise ValueError("ButterflyLens geographic records must be nonempty")
    normalized: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _INPUT_FIELDS:
            raise ValueError("ButterflyLens geographic input fields differ")
        item = dict(record)
        source_id = _text(item["flickr_record_id"], field="flickr_record_id")
        if source_id in normalized:
            raise ValueError("ButterflyLens geographic source identity repeats")
        item["flickr_record_id"] = source_id
        item["geographic_evidence_fingerprint"] = _sha(
            item["geographic_evidence_fingerprint"],
            field="geographic_evidence_fingerprint",
        )
        item["latest_flickr_event_date"] = _optional_date(
            item["latest_flickr_event_date"], field="latest_flickr_event_date"
        )
        _validate_geographic_input(item)
        normalized[source_id] = item
    return normalized


def _validate_geographic_input(item: dict[str, object]) -> None:
    availability = _choice(
        item["geography_availability"],
        _GEOGRAPHY_STATES,
        field="geography_availability",
    )
    public_status = _choice(
        item["public_geometry_status"],
        _PUBLIC_GEOMETRY_STATES,
        field="public_geometry_status",
    )
    item["geography_availability"] = availability
    item["public_geometry_status"] = public_status
    if availability == "h3":
        item["h3_cell"] = _h3(item["h3_cell"])
        item["h3_version"] = _text(item["h3_version"], field="h3_version", maximum=40)
        item["h3_resolution"] = _resolution(
            item["h3_resolution"], field="h3_resolution"
        )
        item["source_precision_metres"] = _nonnegative_number(
            item["source_precision_metres"], field="source_precision_metres"
        )
        item["published_h3_resolution"] = _resolution(
            item["published_h3_resolution"], field="published_h3_resolution"
        )
        if public_status == "withheld" or item["public_geometry_reason"] is not None:
            raise ValueError("ButterflyLens available H3 public geometry differs")
    elif any(
        item[field] is not None
        for field in (
            "h3_cell",
            "h3_version",
            "h3_resolution",
            "source_precision_metres",
            "published_h3_resolution",
        )
    ):
        raise ValueError("ButterflyLens unavailable geography must not invent a cell")
    elif public_status != "withheld":
        raise ValueError("ButterflyLens unavailable geography must be withheld")
    if public_status == "withheld":
        item["public_geometry_reason"] = _text(
            item["public_geometry_reason"], field="public_geometry_reason", maximum=500
        )


def _build_row(
    items: Sequence[Mapping[str, Mapping[str, object]]], *, source_commit: str
) -> dict[str, object]:
    first = items[0]
    geographic = first["geo"]
    evidence = first["evidence"]
    sources = [item["source"] for item in items]
    model_rows = [item["evidence"] for item in items]
    located = geographic["geography_availability"] == "h3"
    project_id = str(evidence["project_id"])
    run_id = str(evidence["run_id"])
    source_snapshots = sorted(
        {str(source["source_snapshot_fingerprint"]) for source in sources}
    )
    source_snapshot = canonical_semantic_fingerprint(
        {"source_snapshot_fingerprints": source_snapshots}
    )
    evidence_fingerprints = sorted(
        {
            *(str(item["geo"]["geographic_evidence_fingerprint"]) for item in items),
            *(str(row["evidence_fingerprint"]) for row in model_rows),
            *source_snapshots,
        }
    )
    candidate_count = len({str(source["flickr_record_id"]) for source in sources})
    latest_dates = sorted(
        str(item["geo"]["latest_flickr_event_date"])
        for item in items
        if item["geo"]["latest_flickr_event_date"] is not None
    )
    unavailable_cell = "candidate evidence has no publishable H3 cell"
    downstream_reason = "ButterflyLens must rebuild and join this evidence downstream"
    body: dict[str, object] = {
        "schema_version": BUTTERFLYLENS_GEOGRAPHIC_IMPACT_VERSION,
        "target_cell_schema_version": BUTTERFLYLENS_TARGET_CELL_VERSION,
        "project_id": project_id,
        "run_id": run_id,
        "snapshot_mode": "submitted",
        "source_commit": source_commit,
        "geography_availability": geographic["geography_availability"],
        "cell_id": geographic["h3_cell"],
        "grid": "H3" if located else None,
        "h3_version": geographic["h3_version"],
        "h3_resolution": geographic["h3_resolution"],
        "accepted_taxon_key": evidence["candidate_accepted_taxon_key"],
        "source_snapshot_fingerprint": source_snapshot,
        "flickr_snapshot_fingerprint": source_snapshot,
        "ala_snapshot_fingerprint": None,
        "provider_union_fingerprint": None,
        "review_projection_fingerprint": None,
        "quality_snapshot_fingerprint": None,
        **_unavailable_count("ala_baseline", downstream_reason),
        **(
            _available_count("flickr_candidate", candidate_count)
            if located
            else _unavailable_count("flickr_candidate", unavailable_cell)
        ),
        **_unavailable_count("yoloe_butterfly", "YOLOE evidence was not joined"),
        **(
            _available_count("bioclip_species_candidate", len(model_rows))
            if located
            else _unavailable_count("bioclip_species_candidate", unavailable_cell)
        ),
        **_unavailable_count("community_reviewed", "review has not occurred"),
        **_unavailable_count(
            "human_supported", "human support has not been established"
        ),
        **_unavailable_count("release_ready", "release gates have not been evaluated"),
        **_unavailable_flag("potential_coverage_gap", downstream_reason),
        **_unavailable_flag(
            "human_supported_additional", "human support is unavailable"
        ),
        **_unavailable_flag("release_ready_additional", "release state is unavailable"),
        "nearest_ala_evidence_distance_m": None,
        "nearest_ala_evidence_distance_state": "unavailable",
        "nearest_ala_evidence_distance_reason": downstream_reason,
        "latest_ala_event_date": None,
        "latest_ala_event_date_state": "unavailable",
        "latest_ala_event_date_reason": downstream_reason,
        "latest_flickr_event_date": latest_dates[-1] if latest_dates else None,
        "latest_flickr_event_date_state": "available"
        if latest_dates
        else "unavailable",
        "latest_flickr_event_date_reason": None
        if latest_dates
        else "event date unavailable",
        "data_deficiency_state": "baseline_unavailable"
        if located
        else "insufficient_precision",
        "data_deficiency_reason": downstream_reason if located else unavailable_cell,
        "public_geometry_status": geographic["public_geometry_status"],
        "source_precision_metres": geographic["source_precision_metres"],
        "published_h3_resolution": geographic["published_h3_resolution"],
        "public_geometry_reason": geographic["public_geometry_reason"],
        "evidence_fingerprints": evidence_fingerprints,
        "candidate_only_is_occurrence": False,
        "no_geo_is_biological_absence": False,
        "database_primary_key_included": False,
        "scientific_claim_allowed": False,
    }
    digest = canonical_semantic_fingerprint(body).removeprefix("sha256:")
    identified = {**body, "geographic_impact_id": f"biominer-impact:{digest}"}
    return {
        **identified,
        "impact_fingerprint": canonical_semantic_fingerprint(identified),
    }


def _validate_row(row: Mapping[str, object]) -> None:
    if (
        row["schema_version"] != BUTTERFLYLENS_GEOGRAPHIC_IMPACT_VERSION
        or row["target_cell_schema_version"] != BUTTERFLYLENS_TARGET_CELL_VERSION
        or row["snapshot_mode"] != "submitted"
        or row["candidate_only_is_occurrence"] is not False
        or row["no_geo_is_biological_absence"] is not False
        or row["database_primary_key_included"] is not False
        or row["scientific_claim_allowed"] is not False
    ):
        raise ValueError("ButterflyLens geographic authority differs")
    validate_git_sha(row["source_commit"], field="source_commit")
    for field in ("project_id", "run_id", "accepted_taxon_key"):
        _text(row[field], field=field)
    _sha(row["source_snapshot_fingerprint"], field="source_snapshot_fingerprint")
    _sha(row["flickr_snapshot_fingerprint"], field="flickr_snapshot_fingerprint")
    if row["source_snapshot_fingerprint"] != row["flickr_snapshot_fingerprint"]:
        raise ValueError("ButterflyLens Flickr snapshot lineage differs")
    if any(
        row[field] is not None
        for field in (
            "ala_snapshot_fingerprint",
            "provider_union_fingerprint",
            "review_projection_fingerprint",
            "quality_snapshot_fingerprint",
        )
    ):
        raise ValueError("ButterflyLens downstream geographic fingerprint was invented")
    availability = _choice(
        row["geography_availability"],
        _GEOGRAPHY_STATES,
        field="geography_availability",
    )
    located = availability == "h3"
    if located:
        _h3(row["cell_id"])
        if row["grid"] != "H3":
            raise ValueError("ButterflyLens geographic grid differs")
        _text(row["h3_version"], field="h3_version", maximum=40)
        _resolution(row["h3_resolution"], field="h3_resolution")
    elif any(
        row[field] is not None
        for field in ("cell_id", "grid", "h3_version", "h3_resolution")
    ):
        raise ValueError("ButterflyLens unavailable geography invented a cell")
    for name in _COUNT_NAMES:
        _validate_state_value(row, name=name, suffix="count", value_type=int)
    for name in _IMPACT_NAMES:
        _validate_state_value(row, name=name, suffix="", value_type=bool)
    if located:
        for name in ("flickr_candidate", "bioclip_species_candidate"):
            if row[f"{name}_count_state"] != "available":
                raise ValueError("ButterflyLens located candidate count differs")
    elif any(
        row[f"{name}_count_state"] == "available"
        for name in ("flickr_candidate", "bioclip_species_candidate")
    ):
        raise ValueError("ButterflyLens no-geo candidate count was fabricated")
    if any(
        row[f"{name}_count_state"] == "available"
        for name in (
            "ala_baseline",
            "yoloe_butterfly",
            "community_reviewed",
            "human_supported",
            "release_ready",
        )
    ):
        raise ValueError("ButterflyLens downstream count was fabricated")
    if any(row[f"{name}_state"] != "unavailable" for name in _IMPACT_NAMES):
        raise ValueError("ButterflyLens downstream impact flag was fabricated")
    if (
        row["nearest_ala_evidence_distance_state"] != "unavailable"
        or row["nearest_ala_evidence_distance_m"] is not None
    ):
        raise ValueError("ButterflyLens ALA distance was fabricated")
    _text(
        row["nearest_ala_evidence_distance_reason"],
        field="nearest_ala_evidence_distance_reason",
        maximum=500,
    )
    if (
        row["latest_ala_event_date_state"] != "unavailable"
        or row["latest_ala_event_date"] is not None
    ):
        raise ValueError("ButterflyLens ALA event date was fabricated")
    _text(
        row["latest_ala_event_date_reason"],
        field="latest_ala_event_date_reason",
        maximum=500,
    )
    latest_flickr = row["latest_flickr_event_date"]
    if latest_flickr is None:
        if row["latest_flickr_event_date_state"] != "unavailable":
            raise ValueError("ButterflyLens Flickr event-date state differs")
        _text(
            row["latest_flickr_event_date_reason"],
            field="latest_flickr_event_date_reason",
            maximum=500,
        )
    elif (
        row["latest_flickr_event_date_state"] != "available"
        or row["latest_flickr_event_date_reason"] is not None
    ):
        raise ValueError("ButterflyLens Flickr event-date evidence differs")
    else:
        _optional_date(latest_flickr, field="latest_flickr_event_date")
    expected_deficiency = (
        "baseline_unavailable" if located else "insufficient_precision"
    )
    if row["data_deficiency_state"] != expected_deficiency:
        raise ValueError("ButterflyLens data-deficiency state differs")
    _text(row["data_deficiency_reason"], field="data_deficiency_reason", maximum=500)
    public_status = _choice(
        row["public_geometry_status"],
        _PUBLIC_GEOMETRY_STATES,
        field="public_geometry_status",
    )
    if located:
        _nonnegative_number(
            row["source_precision_metres"], field="source_precision_metres"
        )
        _resolution(row["published_h3_resolution"], field="published_h3_resolution")
        if public_status == "withheld" or row["public_geometry_reason"] is not None:
            raise ValueError("ButterflyLens public H3 geometry differs")
    elif (
        public_status != "withheld"
        or row["source_precision_metres"] is not None
        or row["published_h3_resolution"] is not None
    ):
        raise ValueError("ButterflyLens unavailable public geometry differs")
    else:
        _text(
            row["public_geometry_reason"],
            field="public_geometry_reason",
            maximum=500,
        )
    parents = _canonical_fingerprints(row["evidence_fingerprints"])
    if list(row["evidence_fingerprints"]) != parents:
        raise ValueError("ButterflyLens geographic evidence lineage is not canonical")
    payload = dict(row)
    fingerprint = payload.pop("impact_fingerprint")
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError("ButterflyLens geographic impact fingerprint differs")
    _sha(fingerprint, field="impact_fingerprint")
    impact_id = payload.pop("geographic_impact_id")
    expected_id = (
        "biominer-impact:"
        f"{canonical_semantic_fingerprint(payload).removeprefix('sha256:')}"
    )
    if impact_id != expected_id:
        raise ValueError("ButterflyLens geographic impact identity differs")


def _validate_state_value(
    row: Mapping[str, object], *, name: str, suffix: str, value_type: type
) -> None:
    base = f"{name}_{suffix}" if suffix else name
    state = row[f"{base}_state"]
    value = row[base]
    reason = row[f"{base}_reason"]
    if state == "available":
        if (
            value is None
            or isinstance(value, bool) != (value_type is bool)
            or not isinstance(value, value_type)
            or reason is not None
        ):
            raise ValueError(f"ButterflyLens available {base} differs")
        if value_type is int and value < 0:
            raise ValueError(f"ButterflyLens available {base} must be nonnegative")
    elif state != "unavailable" or value is not None:
        raise ValueError(f"ButterflyLens unavailable {base} differs")
    else:
        _text(reason, field=f"{base}_reason", maximum=500)


def _available_count(name: str, value: int) -> dict[str, object]:
    return {
        f"{name}_count": value,
        f"{name}_count_state": "available",
        f"{name}_count_reason": None,
    }


def _unavailable_count(name: str, reason: str) -> dict[str, object]:
    return {
        f"{name}_count": None,
        f"{name}_count_state": "unavailable",
        f"{name}_count_reason": reason,
    }


def _unavailable_flag(name: str, reason: str) -> dict[str, object]:
    return {
        name: None,
        f"{name}_state": "unavailable",
        f"{name}_reason": reason,
    }


def _artifact(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    return {
        "role": "geographic_impact",
        "availability": "available",
        "unavailable_reason": None,
        "relative_path": f"artifacts/geographic/{path.name}",
        "media_type": "application/vnd.apache.parquet",
        "schema_version": BUTTERFLYLENS_GEOGRAPHIC_IMPACT_VERSION,
        "semantic_fingerprint": _semantic_fingerprint(frame),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "row_count": frame.height,
        "parent_fingerprints": _parent_fingerprints(frame),
        "evidence_maturity_label": None,
    }


def _semantic_fingerprint(frame: pl.DataFrame) -> str:
    return canonical_semantic_fingerprint(
        {
            "role": "geographic_impact",
            "schema_version": BUTTERFLYLENS_GEOGRAPHIC_IMPACT_VERSION,
            "row_fingerprints": frame["impact_fingerprint"].to_list(),
        }
    )


def _parent_fingerprints(frame: pl.DataFrame) -> list[str]:
    parents = {
        value
        for field in frame.columns
        if field.endswith("fingerprint") or field.endswith("fingerprints")
        for item in frame[field].to_list()
        for value in (item if isinstance(item, list) else [item])
        if isinstance(value, str) and value.startswith("sha256:")
    }
    return sorted(parents - set(frame["impact_fingerprint"].to_list()))


def _canonical_fingerprints(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("evidence_fingerprints must be a sequence")
    values = sorted({_sha(item, field="evidence_fingerprints") for item in value})
    if not values:
        raise ValueError("evidence_fingerprints must be nonempty")
    return values


def _sha(value: object, *, field: str) -> str:
    return validate_fingerprint(value, field=field)


def _text(value: object, *, field: str, maximum: int = 240) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"{field} must be nonblank text")
    return value.strip()


def _choice(value: object, choices: frozenset[str], *, field: str) -> str:
    text = _text(value, field=field)
    if text not in choices:
        raise ValueError(f"unsupported {field}: {text}")
    return text


def _h3(value: object) -> str:
    text = _text(value, field="h3_cell", maximum=15)
    if _H3_PATTERN.fullmatch(text) is None:
        raise ValueError("h3_cell must be a 15-character lowercase H3 identity")
    return text


def _resolution(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 15:
        raise ValueError(f"{field} must be between zero and fifteen")
    return value


def _nonnegative_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return float(value)


def _optional_date(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError(f"{field} must be an ISO date")
    return value


def _wire_sha(value: object) -> str | None:
    return (
        None
        if value is None
        else _sha(value, field="wire_sha256").removeprefix("sha256:")
    )


__all__ = [
    "BUTTERFLYLENS_GEOGRAPHIC_IMPACT_FILE",
    "BUTTERFLYLENS_GEOGRAPHIC_IMPACT_SCHEMA",
    "BUTTERFLYLENS_GEOGRAPHIC_IMPACT_VERSION",
    "ButterflyLensGeographicExport",
    "butterflylens_geographic_cell_documents",
    "build_butterflylens_geographic_impact",
    "export_butterflylens_geographic_impact",
    "validate_butterflylens_geographic_export",
    "validate_butterflylens_geographic_impact",
]
