"""ButterflyLens project, source, media, and raw model-evidence export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import NamedTemporaryFile, mkdtemp

import polars as pl

from biominer.bioclip.dynamic_pool_contracts import (
    validate_dynamic_reference_pool_plans,
)
from biominer.bioclip.dynamic_pool_scores import (
    PROBABILITY_AVAILABILITY_STATES,
    STATISTICAL_SUPPORT_STATES,
    validate_dynamic_pool_candidate_scores,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
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


BUTTERFLYLENS_MODEL_ROLES = (
    "project",
    "run",
    "flickr_source_records",
    "media_objects",
    "model_evidence",
)
BUTTERFLYLENS_PROJECT_PROJECTION_VERSION = (
    "biominer-butterflylens-project-projection-v1.0.0"
)
BUTTERFLYLENS_RUN_PROJECTION_VERSION = "biominer-butterflylens-run-projection-v1.0.0"
BUTTERFLYLENS_FLICKR_SOURCE_VERSION = "biominer-butterflylens-flickr-source-v1.0.0"
BUTTERFLYLENS_MEDIA_OBJECT_VERSION = "biominer-butterflylens-media-object-v1.0.0"
BUTTERFLYLENS_MODEL_EVIDENCE_VERSION = (
    "biominer-butterflylens-model-evidence-projection-v1.0.0"
)
BUTTERFLYLENS_MODEL_EVIDENCE_FILE = "butterflylens_model_evidence.parquet"
PUBLIC_DISCOVERY_CLAIM = (
    "All butterfly candidate images discoverable through the published "
    "ButterflyLens Flickr search plan."
)

_PROJECT_FIELDS = frozenset(
    {
        "schema_version",
        "target_schema_version",
        "project_id",
        "slug",
        "name",
        "description",
        "status",
        "geographic_scope",
        "taxon_scope",
        "discovery_scope",
        "data_policy_version",
        "consent_policy_version",
        "created_at",
        "updated_at",
        "database_primary_key_included",
        "scientific_claim_allowed",
        "project_fingerprint",
    }
)
_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "target_schema_version",
        "run_id",
        "project_id",
        "project_fingerprint",
        "run_kind",
        "mode",
        "status",
        "requested_by",
        "requested_at",
        "started_at",
        "finished_at",
        "updated_at",
        "engine",
        "input_fingerprints",
        "artifact_roles",
        "revision",
        "database_primary_key_included",
        "scientific_claim_allowed",
        "run_fingerprint",
    }
)
_GEOGRAPHIC_SCOPE_FIELDS = frozenset(
    {
        "country_code",
        "boundary_id",
        "boundary_version",
        "boundary_sha256",
        "sensitive_coordinate_policy_version",
    }
)
_TAXON_SCOPE_FIELDS = frozenset({"root_taxon_keys", "taxonomy_fingerprint"})
_DISCOVERY_SCOPE_FIELDS = frozenset(
    {"search_plan_fingerprint", "public_discovery_claim"}
)
_REQUESTED_BY_FIELDS = frozenset({"actor_type", "actor_id"})
_ENGINE_FIELDS = frozenset({"repository", "commit", "interface_version", "command"})
_PROJECT_STATUSES = frozenset({"draft", "active", "paused", "archived"})
_RUN_KINDS = frozenset(
    {
        "taxonomy_pack",
        "ala_baseline",
        "reference_bank",
        "flickr_discovery",
        "vision_pipeline",
        "geographic_impact",
        "quality_snapshot",
        "release_export",
        "full_pipeline",
    }
)
_RUN_MODES = frozenset({"live", "submitted", "replay"})
_SOURCE_INPUT_FIELDS = frozenset(
    {
        "flickr_photo_id",
        "organism_unit_id",
        "source_record_hash",
        "source_snapshot_fingerprint",
        "media_content_sha256",
        "media_byte_count",
        "media_type",
        "decode_status",
        "rights_fingerprint",
        "rights_status",
        "duplicate_group_id",
        "owner_group_id",
        "observation_group_id",
    }
)
_RUN_STATUSES = frozenset(
    {
        "queued",
        "leased",
        "running",
        "paused",
        "cancelling",
        "cancelled",
        "succeeded",
        "failed",
    }
)
_RIGHTS_STATUSES = frozenset(
    {"unknown", "allowed", "blocked", "quarantined", "removed"}
)
_DECODE_STATUSES = frozenset(
    {"pending", "valid", "invalid", "download_failed", "not_applicable"}
)
_SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")

BUTTERFLYLENS_FLICKR_SOURCE_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "project_id": pl.String,
    "run_id": pl.String,
    "flickr_record_id": pl.String,
    "flickr_photo_id": pl.String,
    "organism_unit_id": pl.String,
    "source_record_fingerprint": pl.String,
    "source_snapshot_fingerprint": pl.String,
    "flickr_query_ids": pl.List(pl.String),
    "duplicate_group_id": pl.String,
    "owner_group_id": pl.String,
    "observation_group_id": pl.String,
    "rights_fingerprint": pl.String,
    "discovery_evidence_only": pl.Boolean,
    "human_reviewed": pl.Boolean,
    "occurrence_release_authorized": pl.Boolean,
    "source_row_fingerprint": pl.String,
}

BUTTERFLYLENS_MEDIA_OBJECT_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "project_id": pl.String,
    "run_id": pl.String,
    "media_object_id": pl.String,
    "flickr_record_id": pl.String,
    "flickr_photo_id": pl.String,
    "organism_unit_id": pl.String,
    "visual_input_id": pl.String,
    "visual_input_kind": pl.String,
    "visual_input_version": pl.String,
    "visual_input_contract_fingerprint": pl.String,
    "spatial_crop_applied": pl.Boolean,
    "object_kind": pl.String,
    "media_sha256": pl.String,
    "media_byte_count": pl.UInt64,
    "media_type": pl.String,
    "decode_status": pl.String,
    "rights_fingerprint": pl.String,
    "rights_status": pl.String,
    "duplicate_group_id": pl.String,
    "owner_group_id": pl.String,
    "observation_group_id": pl.String,
    "media_payload_included": pl.Boolean,
    "storage_location_included": pl.Boolean,
    "media_fingerprint": pl.String,
}

BUTTERFLYLENS_MODEL_EVIDENCE_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "project_id": pl.String,
    "run_id": pl.String,
    "model_evidence_id": pl.String,
    "flickr_record_id": pl.String,
    "media_object_id": pl.String,
    "flickr_photo_id": pl.String,
    "organism_unit_id": pl.String,
    "score_id": pl.String,
    "source_score_fingerprint": pl.String,
    "candidate_accepted_taxon_key": pl.String,
    "candidate_scientific_name": pl.String,
    "evidence_kind": pl.String,
    "evidence_status": pl.String,
    "status_reason": pl.String,
    "model_id": pl.String,
    "model_revision": pl.String,
    "model_weights_sha256": pl.String,
    "model_fingerprint": pl.String,
    "preprocessing_fingerprint": pl.String,
    "input_fingerprint": pl.String,
    "output_content_sha256": pl.String,
    "global_raw_component_score": pl.Float64,
    "local_raw_component_score": pl.Float64,
    "family_evidence_raw_score": pl.Float64,
    "fused_raw_score": pl.Float64,
    "candidate_rank": pl.UInt32,
    "margin_to_next_raw": pl.Float64,
    "probability_availability": pl.String,
    "calibrated_probability": pl.Float64,
    "calibrator_fingerprint": pl.String,
    "human_review_required": pl.Boolean,
    "statistical_support_status": pl.String,
    "abstained": pl.Boolean,
    "raw_score_is_probability": pl.Boolean,
    "human_verified": pl.Boolean,
    "occurrence_release_authorized": pl.Boolean,
    "parent_fingerprints": pl.List(pl.String),
    "evidence_fingerprint": pl.String,
}


@dataclass(frozen=True, slots=True)
class ButterflyLensModelLayer:
    flickr_source_records: pl.DataFrame
    media_objects: pl.DataFrame
    model_evidence: pl.DataFrame


@dataclass(frozen=True, slots=True)
class ButterflyLensModelExport:
    root: Path
    model_directory: Path
    artifacts: tuple[dict[str, object], ...]


def build_butterflylens_project_projection(
    *,
    project_id: str,
    slug: str,
    name: str,
    description: str,
    status: str,
    boundary_id: str,
    boundary_version: str,
    boundary_sha256: str,
    sensitive_coordinate_policy_version: str,
    root_taxon_keys: Sequence[str],
    taxonomy_fingerprint: str,
    search_plan_fingerprint: str,
    data_policy_version: str,
    consent_policy_version: str,
    created_at: str | datetime,
    updated_at: str | datetime,
) -> dict[str, object]:
    roots = sorted(
        {_required_text(value, field="root_taxon_keys") for value in root_taxon_keys}
    )
    if not roots:
        raise ValueError("ButterflyLens project root taxon keys must be nonempty")
    body: dict[str, object] = {
        "schema_version": BUTTERFLYLENS_PROJECT_PROJECTION_VERSION,
        "target_schema_version": "butterflylens-project:v1.0.0",
        "project_id": _required_text(project_id, field="project_id"),
        "slug": _slug(slug),
        "name": _bounded_text(name, field="name", maximum=120),
        "description": _bounded_text(
            description, field="description", maximum=2000, allow_empty=True
        ),
        "status": _choice(status, _PROJECT_STATUSES, field="status"),
        "geographic_scope": {
            "country_code": "AU",
            "boundary_id": _required_text(boundary_id, field="boundary_id"),
            "boundary_version": _bounded_text(
                boundary_version, field="boundary_version", maximum=120
            ),
            "boundary_sha256": _sha256(boundary_sha256, field="boundary_sha256"),
            "sensitive_coordinate_policy_version": _bounded_text(
                sensitive_coordinate_policy_version,
                field="sensitive_coordinate_policy_version",
                maximum=120,
            ),
        },
        "taxon_scope": {
            "root_taxon_keys": roots,
            "taxonomy_fingerprint": _sha256(
                taxonomy_fingerprint, field="taxonomy_fingerprint"
            ),
        },
        "discovery_scope": {
            "search_plan_fingerprint": _sha256(
                search_plan_fingerprint, field="search_plan_fingerprint"
            ),
            "public_discovery_claim": PUBLIC_DISCOVERY_CLAIM,
        },
        "data_policy_version": _bounded_text(
            data_policy_version, field="data_policy_version", maximum=120
        ),
        "consent_policy_version": _bounded_text(
            consent_policy_version, field="consent_policy_version", maximum=120
        ),
        "created_at": _utc(created_at),
        "updated_at": _utc(updated_at),
        "database_primary_key_included": False,
        "scientific_claim_allowed": False,
    }
    if body["updated_at"] < body["created_at"]:
        raise ValueError("ButterflyLens project updated_at precedes created_at")
    project = {**body, "project_fingerprint": canonical_semantic_fingerprint(body)}
    validate_butterflylens_project_projection(project)
    return project


def validate_butterflylens_project_projection(project: Mapping[str, object]) -> None:
    if not isinstance(project, Mapping) or set(project) != _PROJECT_FIELDS:
        raise ValueError("ButterflyLens project projection fields differ")
    body = dict(project)
    fingerprint = body.pop("project_fingerprint")
    if fingerprint != canonical_semantic_fingerprint(body):
        raise ValueError("ButterflyLens project projection fingerprint differs")
    if (
        project["schema_version"] != BUTTERFLYLENS_PROJECT_PROJECTION_VERSION
        or project["target_schema_version"] != "butterflylens-project:v1.0.0"
        or project["database_primary_key_included"] is not False
        or project["scientific_claim_allowed"] is not False
    ):
        raise ValueError("ButterflyLens project projection authority differs")
    _required_text(project["project_id"], field="project_id")
    _slug(project["slug"])
    _bounded_text(project["name"], field="name", maximum=120)
    _bounded_text(
        project["description"],
        field="description",
        maximum=2000,
        allow_empty=True,
    )
    _choice(project["status"], _PROJECT_STATUSES, field="status")
    geographic_scope = _exact_mapping(
        project["geographic_scope"],
        fields=_GEOGRAPHIC_SCOPE_FIELDS,
        field="geographic_scope",
    )
    if geographic_scope["country_code"] != "AU":
        raise ValueError("ButterflyLens project country must be AU")
    _required_text(geographic_scope["boundary_id"], field="boundary_id")
    _bounded_text(
        geographic_scope["boundary_version"],
        field="boundary_version",
        maximum=120,
    )
    _sha256(geographic_scope["boundary_sha256"], field="boundary_sha256")
    _bounded_text(
        geographic_scope["sensitive_coordinate_policy_version"],
        field="sensitive_coordinate_policy_version",
        maximum=120,
    )
    taxon_scope = _exact_mapping(
        project["taxon_scope"], fields=_TAXON_SCOPE_FIELDS, field="taxon_scope"
    )
    roots = _canonical_texts(
        taxon_scope["root_taxon_keys"], field="root_taxon_keys", nonempty=True
    )
    if list(taxon_scope["root_taxon_keys"]) != roots:
        raise ValueError("ButterflyLens project root taxon keys are not canonical")
    _sha256(taxon_scope["taxonomy_fingerprint"], field="taxonomy_fingerprint")
    discovery_scope = _exact_mapping(
        project["discovery_scope"],
        fields=_DISCOVERY_SCOPE_FIELDS,
        field="discovery_scope",
    )
    _sha256(
        discovery_scope["search_plan_fingerprint"],
        field="search_plan_fingerprint",
    )
    if discovery_scope["public_discovery_claim"] != PUBLIC_DISCOVERY_CLAIM:
        raise ValueError("ButterflyLens public discovery claim differs")
    _bounded_text(
        project["data_policy_version"], field="data_policy_version", maximum=120
    )
    _bounded_text(
        project["consent_policy_version"],
        field="consent_policy_version",
        maximum=120,
    )
    created = _canonical_utc(project["created_at"], field="created_at")
    updated = _canonical_utc(project["updated_at"], field="updated_at")
    if updated < created:
        raise ValueError("ButterflyLens project updated_at precedes created_at")
    _sha256(fingerprint, field="project_fingerprint")


def build_butterflylens_run_projection(
    *,
    run_id: str,
    project: Mapping[str, object],
    run_kind: str,
    mode: str,
    status: str,
    requested_at: str | datetime,
    started_at: str | datetime | None,
    finished_at: str | datetime | None,
    updated_at: str | datetime,
    producer_commit: str,
    engine_interface_version: str,
    engine_command: str,
    input_fingerprints: Sequence[str],
    revision: int = 1,
) -> dict[str, object]:
    validate_butterflylens_project_projection(project)
    inputs = sorted(
        {_sha256(value, field="input_fingerprints") for value in input_fingerprints}
    )
    if not inputs:
        raise ValueError("ButterflyLens run input fingerprints must be nonempty")
    body: dict[str, object] = {
        "schema_version": BUTTERFLYLENS_RUN_PROJECTION_VERSION,
        "target_schema_version": "butterflylens-run:v1.0.0",
        "run_id": _required_text(run_id, field="run_id"),
        "project_id": project["project_id"],
        "project_fingerprint": project["project_fingerprint"],
        "run_kind": _choice(run_kind, _RUN_KINDS, field="run_kind"),
        "mode": _choice(mode, _RUN_MODES, field="mode"),
        "status": _choice(status, _RUN_STATUSES, field="status"),
        "requested_by": {"actor_type": "system", "actor_id": None},
        "requested_at": _utc(requested_at),
        "started_at": _optional_utc(started_at),
        "finished_at": _optional_utc(finished_at),
        "updated_at": _utc(updated_at),
        "engine": {
            "repository": "karikris/BioMiner",
            "commit": validate_git_sha(producer_commit, field="producer_commit"),
            "interface_version": _bounded_text(
                engine_interface_version,
                field="engine_interface_version",
                maximum=120,
            ),
            "command": _bounded_text(
                engine_command, field="engine_command", maximum=500
            ),
        },
        "input_fingerprints": inputs,
        "artifact_roles": list(BUTTERFLYLENS_MODEL_ROLES),
        "revision": _positive_int(revision, field="revision"),
        "database_primary_key_included": False,
        "scientific_claim_allowed": False,
    }
    _validate_run_timestamps(body)
    run = {**body, "run_fingerprint": canonical_semantic_fingerprint(body)}
    validate_butterflylens_run_projection(run)
    return run


def validate_butterflylens_run_projection(run: Mapping[str, object]) -> None:
    if not isinstance(run, Mapping) or set(run) != _RUN_FIELDS:
        raise ValueError("ButterflyLens run projection fields differ")
    body = dict(run)
    fingerprint = body.pop("run_fingerprint")
    if fingerprint != canonical_semantic_fingerprint(body):
        raise ValueError("ButterflyLens run projection fingerprint differs")
    if (
        run["schema_version"] != BUTTERFLYLENS_RUN_PROJECTION_VERSION
        or run["target_schema_version"] != "butterflylens-run:v1.0.0"
        or run["artifact_roles"] != list(BUTTERFLYLENS_MODEL_ROLES)
        or run["database_primary_key_included"] is not False
        or run["scientific_claim_allowed"] is not False
    ):
        raise ValueError("ButterflyLens run projection authority differs")
    _required_text(run["run_id"], field="run_id")
    _required_text(run["project_id"], field="project_id")
    _sha256(run["project_fingerprint"], field="project_fingerprint")
    _choice(run["run_kind"], _RUN_KINDS, field="run_kind")
    _choice(run["mode"], _RUN_MODES, field="mode")
    _choice(run["status"], _RUN_STATUSES, field="status")
    requested_by = _exact_mapping(
        run["requested_by"], fields=_REQUESTED_BY_FIELDS, field="requested_by"
    )
    if requested_by != {"actor_type": "system", "actor_id": None}:
        raise ValueError("ButterflyLens run requester authority differs")
    engine = _exact_mapping(run["engine"], fields=_ENGINE_FIELDS, field="engine")
    if engine["repository"] != "karikris/BioMiner":
        raise ValueError("ButterflyLens run engine repository differs")
    validate_git_sha(engine["commit"], field="engine.commit")
    _bounded_text(
        engine["interface_version"], field="engine.interface_version", maximum=120
    )
    _bounded_text(engine["command"], field="engine.command", maximum=500)
    inputs = _canonical_fingerprints(
        run["input_fingerprints"], field="input_fingerprints", nonempty=True
    )
    if list(run["input_fingerprints"]) != inputs:
        raise ValueError("ButterflyLens run input fingerprints are not canonical")
    _positive_int(run["revision"], field="revision")
    _validate_run_timestamps(run)
    _sha256(fingerprint, field="run_fingerprint")


def build_butterflylens_model_layer(
    *,
    project: Mapping[str, object],
    run: Mapping[str, object],
    source_media_records: Sequence[Mapping[str, object]],
    candidate_scores: pl.DataFrame,
    pool_plans: pl.DataFrame,
    source_score_artifact_sha256: str,
) -> ButterflyLensModelLayer:
    validate_butterflylens_project_projection(project)
    validate_butterflylens_run_projection(run)
    validate_dynamic_pool_candidate_scores(candidate_scores)
    validate_dynamic_reference_pool_plans(pool_plans)
    if candidate_scores.is_empty():
        raise ValueError("ButterflyLens model export requires candidate scores")
    if (
        run["project_id"] != project["project_id"]
        or run["project_fingerprint"] != project["project_fingerprint"]
    ):
        raise ValueError("ButterflyLens project and run differ")
    if set(candidate_scores["run_id"].to_list()) != {run["run_id"]}:
        raise ValueError("ButterflyLens candidate score run differs")
    if candidate_scores.filter(pl.col("spatial_crop_applied")).height:
        raise ValueError("ButterflyLens model export requires full-frame scores")
    source_output_sha = _sha256(
        source_score_artifact_sha256,
        field="source_score_artifact_sha256",
    )
    source_by_unit = _normalize_source_inputs(source_media_records)
    score_units = {
        (str(row["flickr_photo_id"]), str(row["organism_unit_id"]))
        for row in candidate_scores.iter_rows(named=True)
    }
    if set(source_by_unit) != score_units:
        raise ValueError(
            "ButterflyLens source/media units differ from candidate scores"
        )
    plans = {str(row["plan_id"]): row for row in pool_plans.iter_rows(named=True)}
    if set(candidate_scores["plan_id"].unique().to_list()) - set(plans):
        raise ValueError("ButterflyLens candidate score plan is unavailable")
    source_rows = _source_rows(
        project_id=str(project["project_id"]),
        run_id=str(run["run_id"]),
        source_by_unit=source_by_unit,
        candidate_scores=candidate_scores,
    )
    source_frame = pl.DataFrame(
        source_rows,
        schema=BUTTERFLYLENS_FLICKR_SOURCE_SCHEMA,
        strict=True,
    ).sort("flickr_photo_id", "organism_unit_id")
    source_ids = {
        (str(row["flickr_photo_id"]), str(row["organism_unit_id"])): row
        for row in source_frame.iter_rows(named=True)
    }
    media_rows, media_by_score = _media_rows(
        project_id=str(project["project_id"]),
        run_id=str(run["run_id"]),
        source_by_unit=source_by_unit,
        source_ids=source_ids,
        candidate_scores=candidate_scores,
    )
    media_frame = pl.DataFrame(
        media_rows,
        schema=BUTTERFLYLENS_MEDIA_OBJECT_SCHEMA,
        strict=True,
    ).sort("flickr_photo_id", "organism_unit_id", "visual_input_id")
    evidence_rows = _model_rows(
        project_id=str(project["project_id"]),
        run_id=str(run["run_id"]),
        candidate_scores=candidate_scores,
        plans=plans,
        source_ids=source_ids,
        media_by_score=media_by_score,
        output_content_sha256=source_output_sha,
    )
    model_frame = pl.DataFrame(
        evidence_rows,
        schema=BUTTERFLYLENS_MODEL_EVIDENCE_SCHEMA,
        strict=True,
    ).sort("flickr_photo_id", "organism_unit_id", "candidate_rank", "score_id")
    layer = ButterflyLensModelLayer(source_frame, media_frame, model_frame)
    validate_butterflylens_model_layer(layer)
    return layer


def validate_butterflylens_model_layer(layer: ButterflyLensModelLayer) -> None:
    if not isinstance(layer, ButterflyLensModelLayer):
        raise TypeError("layer must be a ButterflyLensModelLayer")
    frames = (
        (
            layer.flickr_source_records,
            BUTTERFLYLENS_FLICKR_SOURCE_SCHEMA,
            BUTTERFLYLENS_FLICKR_SOURCE_VERSION,
            "source_row_fingerprint",
        ),
        (
            layer.media_objects,
            BUTTERFLYLENS_MEDIA_OBJECT_SCHEMA,
            BUTTERFLYLENS_MEDIA_OBJECT_VERSION,
            "media_fingerprint",
        ),
        (
            layer.model_evidence,
            BUTTERFLYLENS_MODEL_EVIDENCE_SCHEMA,
            BUTTERFLYLENS_MODEL_EVIDENCE_VERSION,
            "evidence_fingerprint",
        ),
    )
    for frame, schema, schema_version, fingerprint_field in frames:
        if frame.schema != schema or frame.is_empty():
            raise ValueError("ButterflyLens model-layer frame schema or rows differ")
        if set(frame["schema_version"].to_list()) != {schema_version}:
            raise ValueError("ButterflyLens model-layer schema version differs")
        if frame[fingerprint_field].n_unique() != frame.height:
            raise ValueError("ButterflyLens model-layer fingerprints repeat")
        for row in frame.iter_rows(named=True):
            payload = dict(row)
            fingerprint = payload.pop(fingerprint_field)
            if fingerprint != canonical_semantic_fingerprint(payload):
                raise ValueError("ButterflyLens model-layer fingerprint differs")
            _sha256(fingerprint, field=fingerprint_field)
    _require_canonical_frame(
        layer.flickr_source_records,
        ("flickr_photo_id", "organism_unit_id"),
        field="Flickr source records",
    )
    _require_canonical_frame(
        layer.media_objects,
        ("flickr_photo_id", "organism_unit_id", "visual_input_id"),
        field="media objects",
    )
    _require_canonical_frame(
        layer.model_evidence,
        ("flickr_photo_id", "organism_unit_id", "candidate_rank", "score_id"),
        field="model evidence",
    )
    scopes = {
        (
            str(row["project_id"]),
            str(row["run_id"]),
        )
        for frame, _, _, _ in frames
        for row in frame.select("project_id", "run_id").unique().iter_rows(named=True)
    }
    if len(scopes) != 1:
        raise ValueError("ButterflyLens model-layer project/run scope differs")
    if layer.flickr_source_records.filter(
        ~pl.col("discovery_evidence_only")
        | pl.col("human_reviewed")
        | pl.col("occurrence_release_authorized")
    ).height:
        raise ValueError("ButterflyLens Flickr source authority differs")
    for row in layer.flickr_source_records.iter_rows(named=True):
        _validate_source_row(row)
    if layer.media_objects.filter(
        pl.col("media_payload_included")
        | pl.col("storage_location_included")
        | pl.col("spatial_crop_applied")
        | (pl.col("visual_input_kind") != "raw_full_image")
        | (pl.col("object_kind") != "full_frame_visual_input")
    ).height:
        raise ValueError("ButterflyLens media export contract differs")
    for row in layer.media_objects.iter_rows(named=True):
        _validate_media_row(row)
    if layer.model_evidence.filter(
        (pl.col("evidence_kind") != "candidate_score")
        | (pl.col("evidence_status") != "completed")
        | pl.col("raw_score_is_probability")
        | pl.col("human_verified")
        | pl.col("occurrence_release_authorized")
    ).height:
        raise ValueError("ButterflyLens model evidence authority differs")
    if layer.model_evidence["score_id"].n_unique() != layer.model_evidence.height:
        raise ValueError("ButterflyLens model score identities repeat")
    for row in layer.model_evidence.iter_rows(named=True):
        _validate_model_row(row)
    source_ids = set(layer.flickr_source_records["flickr_record_id"].to_list())
    media_ids = set(layer.media_objects["media_object_id"].to_list())
    if set(layer.media_objects["flickr_record_id"].to_list()) - source_ids:
        raise ValueError("ButterflyLens media source identity differs")
    if set(layer.model_evidence["flickr_record_id"].to_list()) - source_ids:
        raise ValueError("ButterflyLens model source identity differs")
    if set(layer.model_evidence["media_object_id"].to_list()) - media_ids:
        raise ValueError("ButterflyLens model media identity differs")
    source_by_id = {
        str(row["flickr_record_id"]): row
        for row in layer.flickr_source_records.iter_rows(named=True)
    }
    media_by_id = {
        str(row["media_object_id"]): row
        for row in layer.media_objects.iter_rows(named=True)
    }
    for media in media_by_id.values():
        source = source_by_id[str(media["flickr_record_id"])]
        _require_equal_fields(
            source,
            media,
            (
                "project_id",
                "run_id",
                "flickr_photo_id",
                "organism_unit_id",
                "rights_fingerprint",
                "duplicate_group_id",
                "owner_group_id",
                "observation_group_id",
            ),
            field="source/media",
        )
    for evidence in layer.model_evidence.iter_rows(named=True):
        source = source_by_id[str(evidence["flickr_record_id"])]
        media = media_by_id[str(evidence["media_object_id"])]
        _require_equal_fields(
            source,
            evidence,
            (
                "project_id",
                "run_id",
                "flickr_photo_id",
                "organism_unit_id",
            ),
            field="source/model evidence",
        )
        _require_equal_fields(
            media,
            evidence,
            (
                "project_id",
                "run_id",
                "flickr_record_id",
                "flickr_photo_id",
                "organism_unit_id",
            ),
            field="media/model evidence",
        )
        expected_parents = {
            evidence["source_score_fingerprint"],
            evidence["model_fingerprint"],
            evidence["preprocessing_fingerprint"],
            source["source_row_fingerprint"],
            media["media_fingerprint"],
        }
        if set(evidence["parent_fingerprints"]) != expected_parents:
            raise ValueError("ButterflyLens model evidence lineage differs")


def export_butterflylens_model_evidence(
    *,
    project: Mapping[str, object],
    run: Mapping[str, object],
    layer: ButterflyLensModelLayer,
    output_root: str | Path,
) -> ButterflyLensModelExport:
    validate_butterflylens_project_projection(project)
    validate_butterflylens_run_projection(run)
    validate_butterflylens_model_layer(layer)
    if (
        run["project_id"] != project["project_id"]
        or run["project_fingerprint"] != project["project_fingerprint"]
    ):
        raise ValueError("ButterflyLens project and run differ")
    for frame in (
        layer.flickr_source_records,
        layer.media_objects,
        layer.model_evidence,
    ):
        if set(frame["project_id"].to_list()) != {project["project_id"]} or set(
            frame["run_id"].to_list()
        ) != {run["run_id"]}:
            raise ValueError("ButterflyLens model-layer scope differs")
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("ButterflyLens handoff root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    artifacts_directory = root / "artifacts"
    if artifacts_directory.is_symlink():
        raise ValueError("ButterflyLens artifacts directory must not be a symlink")
    artifacts_directory.mkdir(exist_ok=True)
    model_directory = artifacts_directory / "model"
    if model_directory.exists():
        raise FileExistsError(
            f"ButterflyLens model directory is create-only: {model_directory}"
        )
    staging_root = Path(
        mkdtemp(dir=artifacts_directory, prefix=".butterflylens-model-")
    )
    staging_model = staging_root / "artifacts" / "model"
    staging_model.mkdir(parents=True)
    try:
        descriptors = (
            _write_json_role("project", project, staging_model),
            _write_json_role("run", run, staging_model),
            _write_frame_role(
                "flickr_source_records",
                layer.flickr_source_records,
                "source_row_fingerprint",
                staging_model,
            ),
            _write_frame_role(
                "media_objects",
                layer.media_objects,
                "media_fingerprint",
                staging_model,
            ),
            _write_frame_role(
                "model_evidence",
                layer.model_evidence,
                "evidence_fingerprint",
                staging_model,
            ),
        )
        validate_butterflylens_model_export(staging_root, descriptors)
        staging_model.replace(model_directory)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    result = ButterflyLensModelExport(root, model_directory, descriptors)
    validate_butterflylens_model_export(result.root, result.artifacts)
    return result


def validate_butterflylens_model_export(
    root: str | Path,
    descriptors: Sequence[Mapping[str, object]],
) -> None:
    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError("ButterflyLens handoff root must not be a symlink")
    normalized = normalize_product_artifacts(
        descriptors,
        required_roles=BUTTERFLYLENS_MODEL_ROLES,
        producer_repository="karikris/BioMiner",
        producer_commit="0" * 40,
    )
    by_role = {str(row["role"]): row for row in normalized}
    model_directory = root_path / "artifacts" / "model"
    if model_directory.is_symlink() or not model_directory.is_dir():
        raise ValueError("ButterflyLens model directory is unavailable")
    expected_paths: set[Path] = set()
    values: dict[str, object] = {}
    frame_roles = {
        "flickr_source_records": (
            BUTTERFLYLENS_FLICKR_SOURCE_SCHEMA,
            "source_row_fingerprint",
        ),
        "media_objects": (BUTTERFLYLENS_MEDIA_OBJECT_SCHEMA, "media_fingerprint"),
        "model_evidence": (
            BUTTERFLYLENS_MODEL_EVIDENCE_SCHEMA,
            "evidence_fingerprint",
        ),
    }
    for role in BUTTERFLYLENS_MODEL_ROLES:
        descriptor = by_role[role]
        if descriptor["availability"] != "available":
            raise ValueError(f"ButterflyLens model role {role!r} must be available")
        filename, schema_version = BUTTERFLYLENS_ROLE_DEFAULTS[role]
        if descriptor["schema_version"] != schema_version:
            raise ValueError(f"ButterflyLens model role {role!r} schema differs")
        expected_relative = f"artifacts/model/{filename}"
        if descriptor["relative_path"] != expected_relative:
            raise ValueError(f"ButterflyLens model role {role!r} path differs")
        path = root_path / expected_relative
        expected_paths.add(path.resolve())
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"ButterflyLens model role {role!r} file is unavailable")
        if (
            path.stat().st_size != descriptor["byte_count"]
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ValueError(
                f"ButterflyLens model role {role!r} physical identity differs"
            )
        if role in {"project", "run"}:
            payload = json.loads(path.read_text(encoding="utf-8"))
            (
                validate_butterflylens_project_projection(payload)
                if role == "project"
                else validate_butterflylens_run_projection(payload)
            )
            fingerprint_field = f"{role}_fingerprint"
            if (
                descriptor["row_count"] != 1
                or descriptor["semantic_fingerprint"] != payload[fingerprint_field]
            ):
                raise ValueError(
                    f"ButterflyLens model role {role!r} semantic identity differs"
                )
            if descriptor["parent_fingerprints"] != _json_parents(role, payload):
                raise ValueError(f"ButterflyLens model role {role!r} lineage differs")
            values[role] = payload
        else:
            frame = pl.read_parquet(path)
            schema, fingerprint_field = frame_roles[role]
            if frame.schema != schema:
                raise ValueError(
                    f"ButterflyLens model role {role!r} frame schema differs"
                )
            semantic = _frame_semantic(role, frame, fingerprint_field)
            if (
                frame.height != descriptor["row_count"]
                or semantic != descriptor["semantic_fingerprint"]
            ):
                raise ValueError(
                    f"ButterflyLens model role {role!r} semantic identity differs"
                )
            if descriptor["parent_fingerprints"] != _frame_parents(
                frame, fingerprint_field
            ):
                raise ValueError(f"ButterflyLens model role {role!r} lineage differs")
            values[role] = frame
    layer = ButterflyLensModelLayer(
        values["flickr_source_records"],
        values["media_objects"],
        values["model_evidence"],
    )
    validate_butterflylens_model_layer(layer)
    if (
        values["run"]["project_id"] != values["project"]["project_id"]
        or values["run"]["project_fingerprint"]
        != values["project"]["project_fingerprint"]
    ):
        raise ValueError("ButterflyLens stored project and run differ")
    for frame in (
        values["flickr_source_records"],
        values["media_objects"],
        values["model_evidence"],
    ):
        if set(frame["project_id"].to_list()) != {values["project"]["project_id"]}:
            raise ValueError("ButterflyLens stored project scope differs")
        if set(frame["run_id"].to_list()) != {values["run"]["run_id"]}:
            raise ValueError("ButterflyLens stored run scope differs")
    if by_role["model_evidence"]["evidence_maturity_label"] != "provisional_raw_score":
        raise ValueError("ButterflyLens model evidence maturity differs")
    if any(
        by_role[role]["evidence_maturity_label"] is not None
        for role in BUTTERFLYLENS_MODEL_ROLES
        if role != "model_evidence"
    ):
        raise ValueError("ButterflyLens non-model evidence maturity differs")
    entries = tuple(model_directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("ButterflyLens model directory has unsafe entries")
    if {path.resolve() for path in entries} != expected_paths:
        raise ValueError("ButterflyLens model directory file set differs")


def _source_rows(
    *,
    project_id: str,
    run_id: str,
    source_by_unit: Mapping[tuple[str, str], Mapping[str, object]],
    candidate_scores: pl.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for unit, source in sorted(source_by_unit.items()):
        photo_id, organism_id = unit
        queries = sorted(
            set(
                candidate_scores.filter(
                    (pl.col("flickr_photo_id") == photo_id)
                    & (pl.col("organism_unit_id") == organism_id)
                )["flickr_query_id"].to_list()
            )
        )
        base = {
            "schema_version": BUTTERFLYLENS_FLICKR_SOURCE_VERSION,
            "project_id": project_id,
            "run_id": run_id,
            "flickr_photo_id": photo_id,
            "organism_unit_id": organism_id,
            "source_record_fingerprint": source["source_record_hash"],
            "source_snapshot_fingerprint": source["source_snapshot_fingerprint"],
            "flickr_query_ids": queries,
            "duplicate_group_id": source["duplicate_group_id"],
            "owner_group_id": source["owner_group_id"],
            "observation_group_id": source["observation_group_id"],
            "rights_fingerprint": source["rights_fingerprint"],
            "discovery_evidence_only": True,
            "human_reviewed": False,
            "occurrence_release_authorized": False,
        }
        digest = canonical_semantic_fingerprint(base).removeprefix("sha256:")
        with_id = {**base, "flickr_record_id": f"biominer-flickr-source:{digest}"}
        rows.append(
            {
                **with_id,
                "source_row_fingerprint": canonical_semantic_fingerprint(with_id),
            }
        )
    return rows


def _media_rows(
    *,
    project_id: str,
    run_id: str,
    source_by_unit: Mapping[tuple[str, str], Mapping[str, object]],
    source_ids: Mapping[tuple[str, str], Mapping[str, object]],
    candidate_scores: pl.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, Mapping[str, object]]]:
    rows: list[dict[str, object]] = []
    by_visual: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for score in candidate_scores.iter_rows(named=True):
        unit = (str(score["flickr_photo_id"]), str(score["organism_unit_id"]))
        key = (*unit, str(score["visual_input_id"]))
        if key in by_visual:
            continue
        source = source_by_unit[unit]
        base = {
            "schema_version": BUTTERFLYLENS_MEDIA_OBJECT_VERSION,
            "project_id": project_id,
            "run_id": run_id,
            "flickr_record_id": source_ids[unit]["flickr_record_id"],
            "flickr_photo_id": unit[0],
            "organism_unit_id": unit[1],
            "visual_input_id": score["visual_input_id"],
            "visual_input_kind": score["visual_input_kind"],
            "visual_input_version": score["visual_input_version"],
            "visual_input_contract_fingerprint": score[
                "visual_input_contract_fingerprint"
            ],
            "spatial_crop_applied": score["spatial_crop_applied"],
            "object_kind": "full_frame_visual_input",
            "media_sha256": source["media_content_sha256"],
            "media_byte_count": source["media_byte_count"],
            "media_type": source["media_type"],
            "decode_status": source["decode_status"],
            "rights_fingerprint": source["rights_fingerprint"],
            "rights_status": source["rights_status"],
            "duplicate_group_id": source["duplicate_group_id"],
            "owner_group_id": source["owner_group_id"],
            "observation_group_id": source["observation_group_id"],
            "media_payload_included": False,
            "storage_location_included": False,
        }
        digest = canonical_semantic_fingerprint(base).removeprefix("sha256:")
        with_id = {**base, "media_object_id": f"biominer-media-object:{digest}"}
        complete = {
            **with_id,
            "media_fingerprint": canonical_semantic_fingerprint(with_id),
        }
        rows.append(complete)
        by_visual[key] = complete
    by_score = {
        str(score["score_id"]): by_visual[
            (
                str(score["flickr_photo_id"]),
                str(score["organism_unit_id"]),
                str(score["visual_input_id"]),
            )
        ]
        for score in candidate_scores.iter_rows(named=True)
    }
    return rows, by_score


def _model_rows(
    *,
    project_id: str,
    run_id: str,
    candidate_scores: pl.DataFrame,
    plans: Mapping[str, Mapping[str, object]],
    source_ids: Mapping[tuple[str, str], Mapping[str, object]],
    media_by_score: Mapping[str, Mapping[str, object]],
    output_content_sha256: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for score in candidate_scores.iter_rows(named=True):
        plan = plans[str(score["plan_id"])]
        if score["model_fingerprint"] != plan["model_fingerprint"]:
            raise ValueError("ButterflyLens score and plan model fingerprints differ")
        unit = (str(score["flickr_photo_id"]), str(score["organism_unit_id"]))
        media = media_by_score[str(score["score_id"])]
        input_fingerprint = canonical_semantic_fingerprint(
            {
                "media_fingerprint": media["media_fingerprint"],
                "visual_input_contract_fingerprint": score[
                    "visual_input_contract_fingerprint"
                ],
                "plan_fingerprint": score["plan_fingerprint"],
                "candidate_set_fingerprint": score["candidate_set_fingerprint"],
            }
        )
        parents = sorted(
            {
                score["score_fingerprint"],
                media["media_fingerprint"],
                source_ids[unit]["source_row_fingerprint"],
                plan["model_fingerprint"],
                plan["preprocessing_fingerprint"],
            }
        )
        base = {
            "schema_version": BUTTERFLYLENS_MODEL_EVIDENCE_VERSION,
            "project_id": project_id,
            "run_id": run_id,
            "flickr_record_id": source_ids[unit]["flickr_record_id"],
            "media_object_id": media["media_object_id"],
            "flickr_photo_id": unit[0],
            "organism_unit_id": unit[1],
            "score_id": score["score_id"],
            "source_score_fingerprint": score["score_fingerprint"],
            "candidate_accepted_taxon_key": score["candidate_accepted_taxon_key"],
            "candidate_scientific_name": score["candidate_scientific_name"],
            "evidence_kind": "candidate_score",
            "evidence_status": "completed",
            "status_reason": "",
            "model_id": plan["model_id"],
            "model_revision": plan["model_revision"],
            "model_weights_sha256": plan["model_weights_sha256"],
            "model_fingerprint": plan["model_fingerprint"],
            "preprocessing_fingerprint": plan["preprocessing_fingerprint"],
            "input_fingerprint": input_fingerprint,
            "output_content_sha256": output_content_sha256,
            "global_raw_component_score": score["global_raw_component_score"],
            "local_raw_component_score": score["local_raw_component_score"],
            "family_evidence_raw_score": score["family_evidence_raw_score"],
            "fused_raw_score": score["fused_raw_score"],
            "candidate_rank": score["candidate_rank"],
            "margin_to_next_raw": score["margin_to_next_raw"],
            "probability_availability": score["probability_availability"],
            "calibrated_probability": score["calibrated_probability"],
            "calibrator_fingerprint": score["calibrator_fingerprint"],
            "human_review_required": score["human_review_required"],
            "statistical_support_status": score["statistical_support_status"],
            "abstained": score["abstained"],
            "raw_score_is_probability": False,
            "human_verified": False,
            "occurrence_release_authorized": False,
            "parent_fingerprints": parents,
        }
        digest = canonical_semantic_fingerprint(base).removeprefix("sha256:")
        with_id = {**base, "model_evidence_id": f"biominer-model-evidence:{digest}"}
        rows.append(
            {
                **with_id,
                "evidence_fingerprint": canonical_semantic_fingerprint(with_id),
            }
        )
    return rows


def _validate_source_row(row: Mapping[str, object]) -> None:
    for field in (
        "project_id",
        "run_id",
        "flickr_photo_id",
        "organism_unit_id",
        "duplicate_group_id",
        "owner_group_id",
        "observation_group_id",
    ):
        _required_text(row[field], field=field)
    for field in (
        "source_record_fingerprint",
        "source_snapshot_fingerprint",
        "rights_fingerprint",
    ):
        _sha256(row[field], field=field)
    queries = _canonical_texts(
        row["flickr_query_ids"], field="flickr_query_ids", nonempty=True
    )
    if list(row["flickr_query_ids"]) != queries:
        raise ValueError("ButterflyLens Flickr query identities are not canonical")
    payload = dict(row)
    payload.pop("source_row_fingerprint")
    record_id = payload.pop("flickr_record_id")
    expected_id = (
        "biominer-flickr-source:"
        f"{canonical_semantic_fingerprint(payload).removeprefix('sha256:')}"
    )
    if record_id != expected_id:
        raise ValueError("ButterflyLens Flickr record identity differs")


def _validate_media_row(row: Mapping[str, object]) -> None:
    for field in (
        "project_id",
        "run_id",
        "flickr_record_id",
        "flickr_photo_id",
        "organism_unit_id",
        "visual_input_id",
        "visual_input_kind",
        "visual_input_version",
        "object_kind",
        "media_type",
        "duplicate_group_id",
        "owner_group_id",
        "observation_group_id",
    ):
        _required_text(row[field], field=field)
    for field in (
        "visual_input_contract_fingerprint",
        "media_sha256",
        "rights_fingerprint",
    ):
        _sha256(row[field], field=field)
    _positive_int(row["media_byte_count"], field="media_byte_count")
    _choice(row["decode_status"], _DECODE_STATUSES, field="decode_status")
    _choice(row["rights_status"], _RIGHTS_STATUSES, field="rights_status")
    payload = dict(row)
    payload.pop("media_fingerprint")
    media_id = payload.pop("media_object_id")
    expected_id = (
        "biominer-media-object:"
        f"{canonical_semantic_fingerprint(payload).removeprefix('sha256:')}"
    )
    if media_id != expected_id:
        raise ValueError("ButterflyLens media object identity differs")


def _validate_model_row(row: Mapping[str, object]) -> None:
    for field in (
        "project_id",
        "run_id",
        "flickr_record_id",
        "media_object_id",
        "flickr_photo_id",
        "organism_unit_id",
        "score_id",
        "candidate_accepted_taxon_key",
        "candidate_scientific_name",
        "model_id",
        "model_revision",
    ):
        _required_text(row[field], field=field)
    for field in (
        "source_score_fingerprint",
        "model_weights_sha256",
        "model_fingerprint",
        "preprocessing_fingerprint",
        "input_fingerprint",
        "output_content_sha256",
    ):
        _sha256(row[field], field=field)
    parents = _canonical_fingerprints(
        row["parent_fingerprints"], field="parent_fingerprints", nonempty=True
    )
    if list(row["parent_fingerprints"]) != parents:
        raise ValueError("ButterflyLens model parents are not canonical")
    _positive_int(row["candidate_rank"], field="candidate_rank")
    probability_status = _choice(
        row["probability_availability"],
        PROBABILITY_AVAILABILITY_STATES,
        field="probability_availability",
    )
    probability = row["calibrated_probability"]
    calibrator = row["calibrator_fingerprint"]
    if probability_status == "available":
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0 <= float(probability) <= 1
            or calibrator is None
        ):
            raise ValueError("ButterflyLens available probability evidence differs")
        _sha256(calibrator, field="calibrator_fingerprint")
    elif probability is not None or calibrator is not None:
        raise ValueError("ButterflyLens unavailable probability evidence differs")
    _choice(
        row["statistical_support_status"],
        STATISTICAL_SUPPORT_STATES,
        field="statistical_support_status",
    )
    payload = dict(row)
    payload.pop("evidence_fingerprint")
    evidence_id = payload.pop("model_evidence_id")
    expected_id = (
        "biominer-model-evidence:"
        f"{canonical_semantic_fingerprint(payload).removeprefix('sha256:')}"
    )
    if evidence_id != expected_id:
        raise ValueError("ButterflyLens model evidence identity differs")


def _normalize_source_inputs(
    records: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
    ):
        raise ValueError("ButterflyLens source/media records must be nonempty")
    normalized: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _SOURCE_INPUT_FIELDS:
            raise ValueError("ButterflyLens source/media input fields differ")
        item = dict(record)
        for field in (
            "flickr_photo_id",
            "organism_unit_id",
            "media_type",
            "duplicate_group_id",
            "owner_group_id",
            "observation_group_id",
        ):
            item[field] = _required_text(item[field], field=field)
        for field in (
            "source_record_hash",
            "source_snapshot_fingerprint",
            "media_content_sha256",
            "rights_fingerprint",
        ):
            item[field] = _sha256(item[field], field=field)
        item["media_byte_count"] = _positive_int(
            item["media_byte_count"], field="media_byte_count"
        )
        item["decode_status"] = _choice(
            item["decode_status"], _DECODE_STATUSES, field="decode_status"
        )
        item["rights_status"] = _choice(
            item["rights_status"], _RIGHTS_STATUSES, field="rights_status"
        )
        key = (str(item["flickr_photo_id"]), str(item["organism_unit_id"]))
        if key in normalized:
            raise ValueError("ButterflyLens source/media unit repeats")
        normalized[key] = item
    return normalized


def _write_json_role(
    role: str,
    payload: Mapping[str, object],
    directory: Path,
) -> dict[str, object]:
    filename, schema_version = BUTTERFLYLENS_ROLE_DEFAULTS[role]
    path = directory / filename
    _write_json_create_only(path, payload)
    fingerprint = str(payload[f"{role}_fingerprint"])
    parents = _json_parents(role, payload)
    return _descriptor(role, path, schema_version, fingerprint, 1, parents)


def _write_frame_role(
    role: str,
    frame: pl.DataFrame,
    fingerprint_field: str,
    directory: Path,
) -> dict[str, object]:
    filename, schema_version = BUTTERFLYLENS_ROLE_DEFAULTS[role]
    path = write_parquet(frame, directory / filename, overwrite=False)
    parents = _frame_parents(frame, fingerprint_field)
    return _descriptor(
        role,
        path,
        schema_version,
        _frame_semantic(role, frame, fingerprint_field),
        frame.height,
        parents,
        maturity="provisional_raw_score" if role == "model_evidence" else None,
    )


def _descriptor(
    role: str,
    path: Path,
    schema_version: str,
    semantic_fingerprint: str,
    row_count: int,
    parents: Sequence[str],
    *,
    maturity: str | None = None,
) -> dict[str, object]:
    return {
        "role": role,
        "availability": "available",
        "unavailable_reason": None,
        "relative_path": f"artifacts/model/{path.name}",
        "media_type": "application/json"
        if path.suffix == ".json"
        else "application/vnd.apache.parquet",
        "schema_version": schema_version,
        "semantic_fingerprint": semantic_fingerprint,
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "row_count": row_count,
        "parent_fingerprints": sorted(set(parents)),
        "evidence_maturity_label": maturity,
    }


def _frame_semantic(role: str, frame: pl.DataFrame, field: str) -> str:
    return canonical_semantic_fingerprint(
        {
            "role": role,
            "schema_version": BUTTERFLYLENS_ROLE_DEFAULTS[role][1],
            "row_fingerprints": frame[field].to_list(),
        }
    )


def _json_parents(role: str, payload: Mapping[str, object]) -> list[str]:
    if role == "project":
        return sorted(
            {
                payload["geographic_scope"]["boundary_sha256"],
                payload["taxon_scope"]["taxonomy_fingerprint"],
                payload["discovery_scope"]["search_plan_fingerprint"],
            }
        )
    return sorted({*payload["input_fingerprints"], payload["project_fingerprint"]})


def _frame_parents(frame: pl.DataFrame, fingerprint_field: str) -> list[str]:
    parents: set[str] = set()
    for field in frame.columns:
        if not (
            field.endswith("fingerprint")
            or field.endswith("fingerprints")
            or field.endswith("sha256")
        ):
            continue
        for value in frame[field].to_list():
            candidates = value if isinstance(value, list) else [value]
            parents.update(
                candidate
                for candidate in candidates
                if isinstance(candidate, str) and candidate.startswith("sha256:")
            )
    return sorted(parents - set(frame[fingerprint_field].to_list()))


def _write_json_create_only(path: Path, payload: Mapping[str, object]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(data)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_run_timestamps(run: Mapping[str, object]) -> None:
    requested = _canonical_utc(run["requested_at"], field="requested_at")
    started = _canonical_optional_utc(run["started_at"], field="started_at")
    finished = _canonical_optional_utc(run["finished_at"], field="finished_at")
    updated = _canonical_utc(run["updated_at"], field="updated_at")
    status = _choice(run["status"], _RUN_STATUSES, field="status")
    if updated < requested or (started is not None and started < requested):
        raise ValueError("ButterflyLens run timestamps differ")
    if finished is not None and finished < (started or requested):
        raise ValueError("ButterflyLens run finished_at differs")
    if status == "queued" and (started is not None or finished is not None):
        raise ValueError("ButterflyLens queued run cannot have execution timestamps")
    if status in {"running", "paused", "cancelling"} and (
        started is None or finished is not None
    ):
        raise ValueError("ButterflyLens active run timestamps differ")
    if status in {"cancelled", "succeeded", "failed"} and finished is None:
        raise ValueError("ButterflyLens terminal run requires finished_at")


def _utc(value: object) -> str:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _optional_utc(value: object) -> str | None:
    return None if value is None else _utc(value)


def _canonical_utc(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value != _utc(value):
        raise ValueError(f"{field} must be a canonical UTC instant")
    return value


def _canonical_optional_utc(value: object, *, field: str) -> str | None:
    return None if value is None else _canonical_utc(value, field=field)


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonblank text")
    return value.strip()


def _bounded_text(
    value: object,
    *,
    field: str,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    text = value.strip()
    if (not text and not allow_empty) or len(text) > maximum:
        raise ValueError(f"{field} length is invalid")
    return text


def _sha256(value: object, *, field: str) -> str:
    return validate_fingerprint(value, field=field)


def _choice(value: object, choices: set[str] | frozenset[str], *, field: str) -> str:
    text = _required_text(value, field=field)
    if text not in choices:
        raise ValueError(f"unsupported {field}: {text}")
    return text


def _slug(value: object) -> str:
    text = _required_text(value, field="slug")
    if _SLUG_PATTERN.fullmatch(text) is None or len(text) > 80:
        raise ValueError("ButterflyLens project slug is invalid")
    return text


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _exact_mapping(
    value: object,
    *,
    fields: frozenset[str],
    field: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"ButterflyLens {field} fields differ")
    return value


def _canonical_texts(
    value: object,
    *,
    field: str,
    nonempty: bool,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    values = sorted({_required_text(item, field=field) for item in value})
    if nonempty and not values:
        raise ValueError(f"{field} must be nonempty")
    return values


def _canonical_fingerprints(
    value: object,
    *,
    field: str,
    nonempty: bool,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    values = sorted({_sha256(item, field=field) for item in value})
    if nonempty and not values:
        raise ValueError(f"{field} must be nonempty")
    return values


def _require_canonical_frame(
    frame: pl.DataFrame,
    sort_fields: Sequence[str],
    *,
    field: str,
) -> None:
    if not frame.equals(frame.sort(list(sort_fields))):
        raise ValueError(f"ButterflyLens {field} rows are not canonical")


def _require_equal_fields(
    left: Mapping[str, object],
    right: Mapping[str, object],
    fields: Sequence[str],
    *,
    field: str,
) -> None:
    if any(left[name] != right[name] for name in fields):
        raise ValueError(f"ButterflyLens {field} relationship differs")


__all__ = [
    "BUTTERFLYLENS_FLICKR_SOURCE_SCHEMA",
    "BUTTERFLYLENS_MEDIA_OBJECT_SCHEMA",
    "BUTTERFLYLENS_MODEL_EVIDENCE_FILE",
    "BUTTERFLYLENS_MODEL_EVIDENCE_SCHEMA",
    "BUTTERFLYLENS_MODEL_EVIDENCE_VERSION",
    "BUTTERFLYLENS_MODEL_ROLES",
    "ButterflyLensModelExport",
    "ButterflyLensModelLayer",
    "build_butterflylens_model_layer",
    "build_butterflylens_project_projection",
    "build_butterflylens_run_projection",
    "export_butterflylens_model_evidence",
    "validate_butterflylens_model_export",
    "validate_butterflylens_model_layer",
    "validate_butterflylens_project_projection",
    "validate_butterflylens_run_projection",
]
