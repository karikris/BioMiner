"""Canonical Flickr photo, organism, association, and candidate grains."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.storage.parquet import write_parquet
from biominer.vision.full_frame_attention import (
    TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
    TARGET_FULL_FRAME_PREPROCESSING,
)
from biominer.vision.gates import SUPPORTED_COMPARISON_ROUTES
from biominer.vision.target_full_frame import (
    TARGET_FULL_FRAME_SCORING_UNIT_VERSION,
    TargetFullFramePlan,
)


FLICKR_PHOTO_EMBEDDING_UNIT_SCHEMA_VERSION = "flickr-photo-embedding-unit-v1.0.0"
FLICKR_SCORING_UNIT_SCHEMA_VERSION = "flickr-scoring-unit-v1.0.0"
FLICKR_SCORING_ASSOCIATION_SCHEMA_VERSION = "flickr-scoring-association-v1.0.0"
FLICKR_SCORING_CANDIDATE_SCHEMA_VERSION = "flickr-scoring-candidate-v1.0.0"

FLICKR_PHOTO_EMBEDDING_UNITS_FILE = "flickr_photo_embedding_units.parquet"
FLICKR_SCORING_UNITS_FILE = "flickr_scoring_units.parquet"
FLICKR_SCORING_ASSOCIATIONS_FILE = "flickr_scoring_unit_associations.parquet"
FLICKR_SCORING_CANDIDATES_FILE = "flickr_scoring_unit_candidates.parquet"

FLICKR_SCORING_ASSOCIATION_KINDS = frozenset({"query", "target"})

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PHOTO_UNIT_ID_PATTERN = re.compile(r"flickr-photo-embedding-unit:[0-9a-f]{64}\Z")
_ASSOCIATION_ID_PATTERN = re.compile(r"flickr-scoring-association:[0-9a-f]{64}\Z")
_CANDIDATE_ID_PATTERN = re.compile(r"flickr-scoring-candidate:[0-9a-f]{64}\Z")

_PHOTO_SORT = ("run_id", "source", "flickr_photo_id", "photo_embedding_unit_id")
_SCORING_SORT = (
    "run_id",
    "source",
    "flickr_photo_id",
    "route",
    "organism_unit_id",
)
_ASSOCIATION_SORT = (
    "run_id",
    "source",
    "flickr_photo_id",
    "route",
    "organism_unit_id",
    "association_kind",
    "association_source",
    "association_source_id",
    "association_id",
)
_CANDIDATE_SORT = (
    "run_id",
    "source",
    "flickr_photo_id",
    "route",
    "organism_unit_id",
    "candidate_priority",
    "candidate_accepted_taxon_key",
)

_ASSOCIATION_REQUIRED_INPUT_FIELDS = frozenset(
    {
        "source",
        "flickr_photo_id",
        "association_kind",
        "association_source",
        "association_source_id",
    }
)
_ASSOCIATION_OPTIONAL_INPUT_FIELDS = frozenset(
    {
        "route",
        "flickr_query_id",
        "query_hash",
        "query_tier",
        "search_term",
        "accepted_taxon_key",
        "scientific_name",
    }
)
_CANDIDATE_INPUT_FIELDS = frozenset(
    {
        "organism_unit_id",
        "candidate_accepted_taxon_key",
        "candidate_scientific_name",
        "family_key",
        "family_name",
        "genus_key",
        "genus_name",
        "candidate_priority",
        "candidate_reasons",
        "candidate_source_ids",
    }
)


@dataclass(frozen=True, slots=True)
class FlickrScoringUnitArtifacts:
    """Normalized tables whose foreign keys express all scoring fan-out."""

    photo_embedding_units: pl.DataFrame
    scoring_units: pl.DataFrame
    associations: pl.DataFrame
    candidate_species: pl.DataFrame


def flickr_photo_embedding_unit_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "run_id": pl.String,
        "photo_embedding_unit_id": pl.String,
        "source": pl.String,
        "flickr_photo_id": pl.String,
        "source_record_hash": pl.String,
        "visual_input_id": pl.String,
        "visual_input_kind": pl.String,
        "visual_input_version": pl.String,
        "raw_image_content_hash": pl.String,
        "transformation_fingerprint": pl.String,
        "image_resize_mode": pl.String,
        "preprocessing_contract_fingerprint": pl.String,
        "model_input_signature": pl.String,
        "image_width": pl.UInt32,
        "image_height": pl.UInt32,
        "image_mode": pl.String,
        "photo_unit_fingerprint": pl.String,
    }


def flickr_scoring_unit_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "run_id": pl.String,
        "organism_unit_id": pl.String,
        "photo_embedding_unit_id": pl.String,
        "source": pl.String,
        "flickr_photo_id": pl.String,
        "source_record_hash": pl.String,
        "route": pl.String,
        "visual_input_id": pl.String,
        "model_input_signature": pl.String,
        "routing_scoring_unit_version": pl.String,
        "routing_policy_version": pl.String,
        "routing_policy_fingerprint": pl.String,
        "detection_ids": pl.List(pl.String),
        "detection_count": pl.UInt32,
        "scoring_unit_fingerprint": pl.String,
    }


def flickr_scoring_association_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "run_id": pl.String,
        "association_id": pl.String,
        "organism_unit_id": pl.String,
        "photo_embedding_unit_id": pl.String,
        "source": pl.String,
        "flickr_photo_id": pl.String,
        "route": pl.String,
        "association_kind": pl.String,
        "association_source": pl.String,
        "association_source_id": pl.String,
        "flickr_query_id": pl.String,
        "query_hash": pl.String,
        "query_tier": pl.String,
        "search_term": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "association_fingerprint": pl.String,
    }


def flickr_scoring_candidate_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "run_id": pl.String,
        "candidate_species_id": pl.String,
        "organism_unit_id": pl.String,
        "photo_embedding_unit_id": pl.String,
        "source": pl.String,
        "flickr_photo_id": pl.String,
        "route": pl.String,
        "candidate_accepted_taxon_key": pl.String,
        "candidate_scientific_name": pl.String,
        "family_key": pl.String,
        "family_name": pl.String,
        "genus_key": pl.String,
        "genus_name": pl.String,
        "candidate_priority": pl.UInt32,
        "candidate_reasons": pl.List(pl.String),
        "candidate_source_ids": pl.List(pl.String),
        "candidate_fingerprint": pl.String,
    }


def build_flickr_scoring_unit_artifacts(
    plan: TargetFullFramePlan,
    *,
    run_id: str,
    associations: Sequence[Mapping[str, object]] = (),
    candidate_species: Sequence[Mapping[str, object]] = (),
) -> FlickrScoringUnitArtifacts:
    """Normalize a target-full-frame plan without copying image or vector values."""

    if not isinstance(plan, TargetFullFramePlan):
        raise TypeError("plan must be a TargetFullFramePlan")
    normalized_run_id = _required_text(run_id, field="run_id")
    _require_mapping_sequence(associations, label="Flickr scoring associations")
    _require_mapping_sequence(candidate_species, label="Flickr scoring candidates")

    visual_inputs = {item.visual_input_id: item for item in plan.visual_inputs}
    if len(visual_inputs) != len(plan.visual_inputs):
        raise ValueError("target full-frame plan contains duplicate visual_input_id")

    photo_rows_by_key: dict[tuple[str, str], dict[str, object]] = {}
    scoring_rows: list[dict[str, object]] = []
    for unit in plan.scoring_units:
        if unit.scoring_unit_version != TARGET_FULL_FRAME_SCORING_UNIT_VERSION:
            raise ValueError("unsupported target full-frame scoring-unit version")
        if unit.route not in SUPPORTED_COMPARISON_ROUTES:
            raise ValueError(f"unsupported BioCLIP route {unit.route!r}")
        try:
            visual_input = visual_inputs[unit.raw_visual_input_id]
        except KeyError as exc:
            raise ValueError(
                "target full-frame scoring unit references an unknown visual input"
            ) from exc
        model_input_signature = _model_input_signature(
            visual_input_id=visual_input.visual_input_id,
            visual_input_kind=visual_input.visual_input_kind,
            visual_input_version=visual_input.visual_input_version,
            raw_image_content_hash=visual_input.raw_image_content_hash,
            transformation_fingerprint=visual_input.transformation_fingerprint,
            image_width=visual_input.width,
            image_height=visual_input.height,
            image_mode=visual_input.mode,
        )
        photo_base = {
            "schema_version": FLICKR_PHOTO_EMBEDDING_UNIT_SCHEMA_VERSION,
            "run_id": normalized_run_id,
            "source": _required_text(unit.source, field="source"),
            "flickr_photo_id": _required_text(
                unit.flickr_photo_id, field="flickr_photo_id"
            ),
            "source_record_hash": _sha256(
                unit.source_record_hash, field="source_record_hash"
            ),
            "visual_input_id": _sha256(
                visual_input.visual_input_id, field="visual_input_id"
            ),
            "visual_input_kind": _required_text(
                visual_input.visual_input_kind, field="visual_input_kind"
            ),
            "visual_input_version": _required_text(
                visual_input.visual_input_version, field="visual_input_version"
            ),
            "raw_image_content_hash": _sha256(
                visual_input.raw_image_content_hash,
                field="raw_image_content_hash",
            ),
            "transformation_fingerprint": _sha256(
                visual_input.transformation_fingerprint,
                field="transformation_fingerprint",
            ),
            "image_resize_mode": TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
            "preprocessing_contract_fingerprint": (
                TARGET_FULL_FRAME_PREPROCESSING.fingerprint
            ),
            "model_input_signature": model_input_signature,
            "image_width": _positive_uint32(
                visual_input.width, field="image_width"
            ),
            "image_height": _positive_uint32(
                visual_input.height, field="image_height"
            ),
            "image_mode": _required_text(visual_input.mode, field="image_mode"),
        }
        photo_fingerprint = canonical_semantic_fingerprint(photo_base)
        photo_row = {
            **photo_base,
            "photo_embedding_unit_id": _prefixed_id(
                "flickr-photo-embedding-unit", photo_fingerprint
            ),
            "photo_unit_fingerprint": photo_fingerprint,
        }
        photo_key = (unit.source, unit.flickr_photo_id)
        existing_photo = photo_rows_by_key.setdefault(photo_key, photo_row)
        if existing_photo != photo_row:
            raise ValueError("one Flickr photo maps to conflicting embedding units")

        policy_versions = {
            _required_text(item.routing_policy_version, field="routing_policy_version")
            for item in unit.detections
        }
        policy_fingerprints = {
            _sha256(
                item.routing_policy_fingerprint,
                field="routing_policy_fingerprint",
            )
            for item in unit.detections
        }
        if len(policy_versions) != 1 or len(policy_fingerprints) != 1:
            raise ValueError("one organism unit requires one routing policy identity")
        detection_ids = sorted(
            {_required_text(item.detection_id, field="detection_id") for item in unit.detections}
        )
        if not detection_ids or len(detection_ids) != len(unit.detections):
            raise ValueError("one organism unit requires unique detection evidence")
        scoring_base = {
            "schema_version": FLICKR_SCORING_UNIT_SCHEMA_VERSION,
            "run_id": normalized_run_id,
            "organism_unit_id": _sha256(
                unit.scoring_unit_id, field="organism_unit_id"
            ),
            "photo_embedding_unit_id": photo_row["photo_embedding_unit_id"],
            "source": unit.source,
            "flickr_photo_id": unit.flickr_photo_id,
            "source_record_hash": unit.source_record_hash,
            "route": unit.route,
            "visual_input_id": visual_input.visual_input_id,
            "model_input_signature": model_input_signature,
            "routing_scoring_unit_version": unit.scoring_unit_version,
            "routing_policy_version": next(iter(policy_versions)),
            "routing_policy_fingerprint": next(iter(policy_fingerprints)),
            "detection_ids": detection_ids,
            "detection_count": len(detection_ids),
        }
        scoring_rows.append(
            {
                **scoring_base,
                "scoring_unit_fingerprint": canonical_semantic_fingerprint(
                    scoring_base
                ),
            }
        )

    photo_frame = _frame(
        list(photo_rows_by_key.values()),
        schema=flickr_photo_embedding_unit_schema(),
        sort=_PHOTO_SORT,
    )
    scoring_frame = _frame(
        scoring_rows,
        schema=flickr_scoring_unit_schema(),
        sort=_SCORING_SORT,
    )
    scoring_by_id = {
        str(row["organism_unit_id"]): row
        for row in scoring_frame.iter_rows(named=True)
    }
    units_by_photo: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in scoring_by_id.values():
        units_by_photo.setdefault(
            (str(row["source"]), str(row["flickr_photo_id"])), []
        ).append(row)

    association_rows = _build_association_rows(
        associations,
        run_id=normalized_run_id,
        units_by_photo=units_by_photo,
    )
    candidate_rows = _build_candidate_rows(
        candidate_species,
        run_id=normalized_run_id,
        scoring_by_id=scoring_by_id,
    )
    artifacts = FlickrScoringUnitArtifacts(
        photo_embedding_units=photo_frame,
        scoring_units=scoring_frame,
        associations=_frame(
            association_rows,
            schema=flickr_scoring_association_schema(),
            sort=_ASSOCIATION_SORT,
        ),
        candidate_species=_frame(
            candidate_rows,
            schema=flickr_scoring_candidate_schema(),
            sort=_CANDIDATE_SORT,
        ),
    )
    validate_flickr_scoring_unit_artifacts(artifacts)
    return artifacts


def validate_flickr_scoring_unit_artifacts(
    artifacts: FlickrScoringUnitArtifacts,
) -> None:
    if not isinstance(artifacts, FlickrScoringUnitArtifacts):
        raise TypeError("artifacts must be FlickrScoringUnitArtifacts")
    _validate_frame(
        artifacts.photo_embedding_units,
        schema=flickr_photo_embedding_unit_schema(),
        sort=_PHOTO_SORT,
        label="photo embedding units",
    )
    _validate_frame(
        artifacts.scoring_units,
        schema=flickr_scoring_unit_schema(),
        sort=_SCORING_SORT,
        label="scoring units",
    )
    _validate_frame(
        artifacts.associations,
        schema=flickr_scoring_association_schema(),
        sort=_ASSOCIATION_SORT,
        label="scoring associations",
    )
    _validate_frame(
        artifacts.candidate_species,
        schema=flickr_scoring_candidate_schema(),
        sort=_CANDIDATE_SORT,
        label="scoring candidates",
    )

    photos = artifacts.photo_embedding_units
    scoring = artifacts.scoring_units
    associations = artifacts.associations
    candidates = artifacts.candidate_species
    _require_unique(photos, ("run_id", "source", "flickr_photo_id"), label="photo grain")
    _require_unique(photos, ("photo_embedding_unit_id",), label="photo unit identity")
    _require_unique(
        scoring,
        ("run_id", "source", "flickr_photo_id", "route"),
        label="organism/routing grain",
    )
    _require_unique(scoring, ("organism_unit_id",), label="organism unit identity")
    _require_unique(associations, ("association_id",), label="association identity")
    _require_unique(
        associations,
        (
            "organism_unit_id",
            "association_kind",
            "association_source",
            "association_source_id",
        ),
        label="organism association grain",
    )
    _require_unique(candidates, ("candidate_species_id",), label="candidate identity")
    _require_unique(
        candidates,
        ("organism_unit_id", "candidate_accepted_taxon_key"),
        label="organism candidate-species grain",
    )

    photo_by_id = {
        str(row["photo_embedding_unit_id"]): row
        for row in photos.iter_rows(named=True)
    }
    scoring_by_id = {
        str(row["organism_unit_id"]): row for row in scoring.iter_rows(named=True)
    }
    for row in photos.iter_rows(named=True):
        _validate_photo_row(row)
    for row in scoring.iter_rows(named=True):
        _validate_scoring_row(row, photo_by_id=photo_by_id)
    for row in associations.iter_rows(named=True):
        _validate_association_row(row, scoring_by_id=scoring_by_id)
    for row in candidates.iter_rows(named=True):
        _validate_candidate_row(row, scoring_by_id=scoring_by_id)

    if scoring.height and scoring["photo_embedding_unit_id"].n_unique() != photos.height:
        raise ValueError("every photo embedding unit must be referenced by scoring work")
    signature_counts = photos.group_by("visual_input_id").agg(
        pl.col("model_input_signature").n_unique().alias("signature_count")
    )
    if signature_counts.filter(pl.col("signature_count") != 1).height:
        raise ValueError("one visual input maps to conflicting model-input signatures")


def write_flickr_scoring_unit_artifacts(
    artifacts: FlickrScoringUnitArtifacts,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_flickr_scoring_unit_artifacts(artifacts)
    output = Path(output_dir)
    return {
        "photo_embedding_units": write_parquet(
            artifacts.photo_embedding_units,
            output / FLICKR_PHOTO_EMBEDDING_UNITS_FILE,
        ),
        "scoring_units": write_parquet(
            artifacts.scoring_units,
            output / FLICKR_SCORING_UNITS_FILE,
        ),
        "associations": write_parquet(
            artifacts.associations,
            output / FLICKR_SCORING_ASSOCIATIONS_FILE,
        ),
        "candidate_species": write_parquet(
            artifacts.candidate_species,
            output / FLICKR_SCORING_CANDIDATES_FILE,
        ),
    }


def _build_association_rows(
    values: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    units_by_photo: Mapping[tuple[str, str], Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for raw in values:
        missing = sorted(_ASSOCIATION_REQUIRED_INPUT_FIELDS - set(raw))
        extra = sorted(
            set(raw)
            - _ASSOCIATION_REQUIRED_INPUT_FIELDS
            - _ASSOCIATION_OPTIONAL_INPUT_FIELDS
        )
        if missing or extra:
            raise ValueError(
                f"Flickr scoring association fields mismatch: missing={missing}, extra={extra}"
            )
        source = _required_text(raw["source"], field="source")
        photo_id = _required_text(raw["flickr_photo_id"], field="flickr_photo_id")
        kind = _required_text(raw["association_kind"], field="association_kind").casefold()
        if kind not in FLICKR_SCORING_ASSOCIATION_KINDS:
            raise ValueError(f"unsupported scoring association kind {kind!r}")
        route = _optional_text(raw.get("route"), field="route")
        if route is not None and route not in SUPPORTED_COMPARISON_ROUTES:
            raise ValueError(f"unsupported association route {route!r}")
        query_id = _optional_text(raw.get("flickr_query_id"), field="flickr_query_id")
        query_hash = _optional_text(raw.get("query_hash"), field="query_hash")
        query_tier = _optional_text(raw.get("query_tier"), field="query_tier")
        search_term = _optional_text(raw.get("search_term"), field="search_term")
        taxon_key = _optional_text(raw.get("accepted_taxon_key"), field="accepted_taxon_key")
        scientific_name = _optional_text(raw.get("scientific_name"), field="scientific_name")
        if (taxon_key is None) != (scientific_name is None):
            raise ValueError("association taxonomy requires both key and scientific name")
        if kind == "query" and query_id is None and query_hash is None:
            raise ValueError("query association requires flickr_query_id or query_hash")
        if kind == "target" and taxon_key is None:
            raise ValueError("target association requires accepted taxonomy")
        matching = list(units_by_photo.get((source, photo_id), ()))
        if route is not None:
            matching = [row for row in matching if row["route"] == route]
        if not matching:
            raise ValueError("association does not match an eligible Flickr scoring unit")
        for unit in matching:
            base = {
                "schema_version": FLICKR_SCORING_ASSOCIATION_SCHEMA_VERSION,
                "run_id": run_id,
                "organism_unit_id": unit["organism_unit_id"],
                "photo_embedding_unit_id": unit["photo_embedding_unit_id"],
                "source": source,
                "flickr_photo_id": photo_id,
                "route": unit["route"],
                "association_kind": kind,
                "association_source": _required_text(
                    raw["association_source"], field="association_source"
                ),
                "association_source_id": _required_text(
                    raw["association_source_id"], field="association_source_id"
                ),
                "flickr_query_id": query_id,
                "query_hash": query_hash,
                "query_tier": query_tier,
                "search_term": search_term,
                "accepted_taxon_key": taxon_key,
                "scientific_name": scientific_name,
            }
            fingerprint = canonical_semantic_fingerprint(base)
            output.append(
                {
                    **base,
                    "association_id": _prefixed_id(
                        "flickr-scoring-association", fingerprint
                    ),
                    "association_fingerprint": fingerprint,
                }
            )
    return output


def _build_candidate_rows(
    values: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    scoring_by_id: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for raw in values:
        if set(raw) != _CANDIDATE_INPUT_FIELDS:
            raise ValueError(
                "Flickr scoring candidate fields mismatch: "
                f"missing={sorted(_CANDIDATE_INPUT_FIELDS - set(raw))}, "
                f"extra={sorted(set(raw) - _CANDIDATE_INPUT_FIELDS)}"
            )
        organism_unit_id = _sha256(
            raw["organism_unit_id"], field="organism_unit_id"
        )
        try:
            unit = scoring_by_id[organism_unit_id]
        except KeyError as exc:
            raise ValueError(
                "candidate species references an unknown organism unit"
            ) from exc
        base = {
            "schema_version": FLICKR_SCORING_CANDIDATE_SCHEMA_VERSION,
            "run_id": run_id,
            "organism_unit_id": organism_unit_id,
            "photo_embedding_unit_id": unit["photo_embedding_unit_id"],
            "source": unit["source"],
            "flickr_photo_id": unit["flickr_photo_id"],
            "route": unit["route"],
            "candidate_accepted_taxon_key": _required_text(
                raw["candidate_accepted_taxon_key"],
                field="candidate_accepted_taxon_key",
            ),
            "candidate_scientific_name": _required_text(
                raw["candidate_scientific_name"],
                field="candidate_scientific_name",
            ),
            "family_key": _required_text(raw["family_key"], field="family_key"),
            "family_name": _required_text(raw["family_name"], field="family_name"),
            "genus_key": _required_text(raw["genus_key"], field="genus_key"),
            "genus_name": _required_text(raw["genus_name"], field="genus_name"),
            "candidate_priority": _nonnegative_uint32(
                raw["candidate_priority"], field="candidate_priority"
            ),
            "candidate_reasons": _canonical_strings(
                raw["candidate_reasons"], field="candidate_reasons"
            ),
            "candidate_source_ids": _canonical_strings(
                raw["candidate_source_ids"], field="candidate_source_ids"
            ),
        }
        fingerprint = canonical_semantic_fingerprint(base)
        output.append(
            {
                **base,
                "candidate_species_id": _prefixed_id(
                    "flickr-scoring-candidate", fingerprint
                ),
                "candidate_fingerprint": fingerprint,
            }
        )
    return output


def _validate_photo_row(row: Mapping[str, object]) -> None:
    if row["schema_version"] != FLICKR_PHOTO_EMBEDDING_UNIT_SCHEMA_VERSION:
        raise ValueError("unsupported photo embedding-unit schema version")
    _sha256(row["source_record_hash"], field="source_record_hash")
    _sha256(row["visual_input_id"], field="visual_input_id")
    _sha256(row["raw_image_content_hash"], field="raw_image_content_hash")
    _sha256(row["transformation_fingerprint"], field="transformation_fingerprint")
    _sha256(
        row["preprocessing_contract_fingerprint"],
        field="preprocessing_contract_fingerprint",
    )
    expected_signature = _model_input_signature(
        visual_input_id=str(row["visual_input_id"]),
        visual_input_kind=str(row["visual_input_kind"]),
        visual_input_version=str(row["visual_input_version"]),
        raw_image_content_hash=str(row["raw_image_content_hash"]),
        transformation_fingerprint=str(row["transformation_fingerprint"]),
        image_width=int(row["image_width"]),
        image_height=int(row["image_height"]),
        image_mode=str(row["image_mode"]),
    )
    if row["model_input_signature"] != expected_signature:
        raise ValueError("photo model-input signature mismatch")
    base = _without(row, "photo_embedding_unit_id", "photo_unit_fingerprint")
    fingerprint = canonical_semantic_fingerprint(base)
    if row["photo_unit_fingerprint"] != fingerprint:
        raise ValueError("photo embedding-unit fingerprint mismatch")
    expected_id = _prefixed_id("flickr-photo-embedding-unit", fingerprint)
    if row["photo_embedding_unit_id"] != expected_id:
        raise ValueError("photo embedding-unit identity mismatch")
    if not _PHOTO_UNIT_ID_PATTERN.fullmatch(str(row["photo_embedding_unit_id"])):
        raise ValueError("photo embedding-unit ID is invalid")


def _validate_scoring_row(
    row: Mapping[str, object],
    *,
    photo_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    if row["schema_version"] != FLICKR_SCORING_UNIT_SCHEMA_VERSION:
        raise ValueError("unsupported Flickr scoring-unit schema version")
    _sha256(row["organism_unit_id"], field="organism_unit_id")
    _sha256(row["scoring_unit_fingerprint"], field="scoring_unit_fingerprint")
    if row["route"] not in SUPPORTED_COMPARISON_ROUTES:
        raise ValueError("unsupported Flickr scoring-unit route")
    detection_ids = list(row["detection_ids"])
    if detection_ids != sorted(set(detection_ids)) or not detection_ids:
        raise ValueError("scoring-unit detection IDs must be sorted and unique")
    if row["detection_count"] != len(detection_ids):
        raise ValueError("scoring-unit detection count mismatch")
    try:
        photo = photo_by_id[str(row["photo_embedding_unit_id"])]
    except KeyError as exc:
        raise ValueError("scoring unit references an unknown photo unit") from exc
    for field in (
        "run_id",
        "source",
        "flickr_photo_id",
        "source_record_hash",
        "visual_input_id",
        "model_input_signature",
    ):
        if row[field] != photo[field]:
            raise ValueError(f"scoring/photo unit {field} mismatch")
    expected = canonical_semantic_fingerprint(
        _without(row, "scoring_unit_fingerprint")
    )
    if row["scoring_unit_fingerprint"] != expected:
        raise ValueError("Flickr scoring-unit fingerprint mismatch")


def _validate_association_row(
    row: Mapping[str, object],
    *,
    scoring_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    if row["schema_version"] != FLICKR_SCORING_ASSOCIATION_SCHEMA_VERSION:
        raise ValueError("unsupported scoring-association schema version")
    if row["association_kind"] not in FLICKR_SCORING_ASSOCIATION_KINDS:
        raise ValueError("unsupported scoring-association kind")
    _validate_child_unit_identity(row, scoring_by_id=scoring_by_id)
    if row["association_kind"] == "query" and not (
        row["flickr_query_id"] or row["query_hash"]
    ):
        raise ValueError("query association lost its query identity")
    if row["association_kind"] == "target" and not row["accepted_taxon_key"]:
        raise ValueError("target association lost its accepted taxonomy")
    if bool(row["accepted_taxon_key"]) != bool(row["scientific_name"]):
        raise ValueError("association taxonomy is incomplete")
    base = _without(row, "association_id", "association_fingerprint")
    fingerprint = canonical_semantic_fingerprint(base)
    if row["association_fingerprint"] != fingerprint:
        raise ValueError("scoring-association fingerprint mismatch")
    if row["association_id"] != _prefixed_id(
        "flickr-scoring-association", fingerprint
    ):
        raise ValueError("scoring-association identity mismatch")
    if not _ASSOCIATION_ID_PATTERN.fullmatch(str(row["association_id"])):
        raise ValueError("scoring-association ID is invalid")


def _validate_candidate_row(
    row: Mapping[str, object],
    *,
    scoring_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    if row["schema_version"] != FLICKR_SCORING_CANDIDATE_SCHEMA_VERSION:
        raise ValueError("unsupported scoring-candidate schema version")
    _validate_child_unit_identity(row, scoring_by_id=scoring_by_id)
    reasons = list(row["candidate_reasons"])
    sources = list(row["candidate_source_ids"])
    if reasons != sorted(set(reasons)) or not reasons:
        raise ValueError("candidate reasons must be non-empty, sorted and unique")
    if sources != sorted(set(sources)) or not sources:
        raise ValueError("candidate source IDs must be non-empty, sorted and unique")
    base = _without(row, "candidate_species_id", "candidate_fingerprint")
    fingerprint = canonical_semantic_fingerprint(base)
    if row["candidate_fingerprint"] != fingerprint:
        raise ValueError("scoring-candidate fingerprint mismatch")
    if row["candidate_species_id"] != _prefixed_id(
        "flickr-scoring-candidate", fingerprint
    ):
        raise ValueError("scoring-candidate identity mismatch")
    if not _CANDIDATE_ID_PATTERN.fullmatch(str(row["candidate_species_id"])):
        raise ValueError("scoring-candidate ID is invalid")


def _validate_child_unit_identity(
    row: Mapping[str, object],
    *,
    scoring_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    try:
        unit = scoring_by_id[str(row["organism_unit_id"])]
    except KeyError as exc:
        raise ValueError("child row references an unknown organism unit") from exc
    for field in (
        "run_id",
        "photo_embedding_unit_id",
        "source",
        "flickr_photo_id",
        "route",
    ):
        if row[field] != unit[field]:
            raise ValueError(f"child/scoring unit {field} mismatch")


def _model_input_signature(
    *,
    visual_input_id: str,
    visual_input_kind: str,
    visual_input_version: str,
    raw_image_content_hash: str,
    transformation_fingerprint: str,
    image_width: int,
    image_height: int,
    image_mode: str,
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": "flickr-model-input-signature-v1",
            "visual_input_id": visual_input_id,
            "visual_input_kind": visual_input_kind,
            "visual_input_version": visual_input_version,
            "raw_image_content_hash": raw_image_content_hash,
            "transformation_fingerprint": transformation_fingerprint,
            "image_resize_mode": TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
            "preprocessing_contract_fingerprint": (
                TARGET_FULL_FRAME_PREPROCESSING.fingerprint
            ),
            "image_width": image_width,
            "image_height": image_height,
            "image_mode": image_mode,
        }
    )


def _frame(
    rows: Sequence[Mapping[str, object]],
    *,
    schema: dict[str, pl.DataType],
    sort: Sequence[str],
) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row", strict=True).sort(*sort)


def _validate_frame(
    frame: pl.DataFrame,
    *,
    schema: dict[str, pl.DataType],
    sort: Sequence[str],
    label: str,
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{label} must be a Polars DataFrame")
    if frame.schema != schema:
        raise ValueError(f"{label} schema mismatch")
    if not frame.equals(frame.sort(*sort)):
        raise ValueError(f"{label} are not canonically sorted")


def _require_unique(
    frame: pl.DataFrame,
    fields: Sequence[str],
    *,
    label: str,
) -> None:
    if not frame.is_empty() and frame.select(*fields).n_unique() != frame.height:
        raise ValueError(f"{label} is not unique")


def _require_mapping_sequence(
    values: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> None:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    if any(not isinstance(value, Mapping) for value in values):
        raise TypeError(f"{label} must contain mappings")


def _without(row: Mapping[str, object], *fields: str) -> dict[str, object]:
    excluded = set(fields)
    return {key: value for key, value in row.items() if key not in excluded}


def _prefixed_id(prefix: str, fingerprint: str) -> str:
    return f"{prefix}:{fingerprint.removeprefix('sha256:')}"


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} cannot be blank")
    return text


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase sha256 fingerprint")
    return text


def _positive_uint32(value: object, *, field: str) -> int:
    normalized = _nonnegative_uint32(value, field=field)
    if normalized == 0:
        raise ValueError(f"{field} must be positive")
    return normalized


def _nonnegative_uint32(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if normalized < 0 or normalized > 2**32 - 1:
        raise ValueError(f"{field} is outside UInt32 range")
    return normalized


def _canonical_strings(value: object, *, field: str) -> list[str]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    normalized = sorted({_required_text(item, field=field) for item in value})
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


__all__ = [
    "FLICKR_PHOTO_EMBEDDING_UNITS_FILE",
    "FLICKR_PHOTO_EMBEDDING_UNIT_SCHEMA_VERSION",
    "FLICKR_SCORING_ASSOCIATIONS_FILE",
    "FLICKR_SCORING_ASSOCIATION_KINDS",
    "FLICKR_SCORING_ASSOCIATION_SCHEMA_VERSION",
    "FLICKR_SCORING_CANDIDATES_FILE",
    "FLICKR_SCORING_CANDIDATE_SCHEMA_VERSION",
    "FLICKR_SCORING_UNITS_FILE",
    "FLICKR_SCORING_UNIT_SCHEMA_VERSION",
    "FlickrScoringUnitArtifacts",
    "build_flickr_scoring_unit_artifacts",
    "flickr_photo_embedding_unit_schema",
    "flickr_scoring_association_schema",
    "flickr_scoring_candidate_schema",
    "flickr_scoring_unit_schema",
    "validate_flickr_scoring_unit_artifacts",
    "write_flickr_scoring_unit_artifacts",
]
